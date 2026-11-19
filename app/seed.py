"""
seed.py — populate the support DB with deterministic fake data.

Two kinds of data are created:

1. **Curated scenarios** for customers 1 and 2. These are the cases the
   agent's guardrails must handle, so they are pinned to exact dates relative
   to "now" and never randomised:

     Customer 1 (Priya — the default demo login)
       A. delivered 5 days ago, small items        -> refund within agent limit
       B. delivered 10 days ago, Rs. 4,499 shoes   -> refund > Rs. 2,000, must escalate
       C. delivered 45 days ago                    -> outside 30-day window
       D. shipped, not delivered                   -> cannot cancel, cannot return yet
       E. processing                               -> cancellable
       F. delivered 3 days ago, non-returnable     -> return must be refused
       G. cancelled                                -> nothing to do

     Customer 2 (Rahul)
       H. delivered 8 days ago + an existing 'requested' return
       I. delivered 15 days ago + a high-value return that was escalated
       -> Rahul's order IDs are what you use to test the ownership guardrail
          while logged in as Priya.

2. **Background noise** for customers 3..N: random orders so list/search tools
   have something realistic to page through.

Randomness uses a seeded `random.Random`, so every run produces identical IDs
and amounts — useful for writing tests against known order numbers.

Usage:  python seed.py            (wipes and reseeds support.db)
        python seed.py --db x.db
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from db import DB_PATH, format_paise, get_db, init_db, to_db_ts, utc_now

# Mirrors the refund guardrail (Rs. 2,000 = 200,000 paise). Kept local so the
# seed script has no dependency on guardrails.py; guardrails.py owns the
# canonical value used at runtime.
REFUND_ESCALATION_THRESHOLD_PAISE: int = 200_000

# Shipping rules from policy.txt §1, in paise.
FREE_SHIPPING_MIN_PAISE: int = 99_900
FLAT_SHIPPING_PAISE: int = 7_900

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]


# --------------------------------------------------------------------------- #
# Static catalogue data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProductSeed:
    sku: str
    name: str
    category: str
    price_paise: int
    stock_quantity: int
    is_returnable: bool = True


PRODUCTS: list[ProductSeed] = [
    # footwear
    ProductSeed("FW-RUN-001", "Stride Runner Running Shoes", "footwear", 449_900, 25),
    ProductSeed("FW-SND-002", "Coastline Sandals", "footwear", 129_900, 40),
    ProductSeed("FW-SNK-003", "Urban Canvas Sneakers", "footwear", 189_900, 30),
    # apparel
    ProductSeed("AP-TSH-001", "Cotton Crew T-Shirt", "apparel", 59_900, 120),
    ProductSeed("AP-JNS-002", "Slim Fit Denim Jeans", "apparel", 179_900, 60),
    ProductSeed("AP-JKT-003", "Monsoon Rain Jacket", "apparel", 249_900, 18),
    ProductSeed("AP-INN-004", "Cotton Briefs (Pack of 3)", "apparel", 49_900, 200, False),
    # accessories
    ProductSeed("AC-WLT-001", "Leather Bifold Wallet", "accessories", 89_900, 50),
    ProductSeed("AC-BAG-002", "Everyday Backpack 25L", "accessories", 219_900, 22),
    ProductSeed("AC-GFT-003", "Gift Card Rs. 1,000", "accessories", 100_000, 999, False),
    # electronics
    ProductSeed("EL-EAR-001", "Wireless Earbuds Pro", "electronics", 299_900, 15),
    ProductSeed("EL-PWR-002", "10000mAh Power Bank", "electronics", 149_900, 35),
    ProductSeed("EL-CBL-003", "USB-C Cable 1m", "electronics", 29_900, 300),
    # personal care (non-returnable once opened)
    ProductSeed("PC-SRM-001", "Vitamin C Face Serum 30ml", "personal-care", 69_900, 80, False),
]


@dataclass(frozen=True)
class CustomerSeed:
    name: str
    email: str
    phone: str
    address_line: str
    city: str
    state: str
    pincode: str

    @property
    def shipping_address(self) -> str:
        return f"{self.address_line}, {self.city}, {self.state} - {self.pincode}"


# A mix of metro, tier-2 and remote addresses so delivery-time answers vary.
CUSTOMERS: list[CustomerSeed] = [
    CustomerSeed("Priya Sharma", "priya.sharma@example.com", "+91 98300 11111",
                 "12B Lake View Road", "Kolkata", "West Bengal", "700029"),
    CustomerSeed("Rahul Verma", "rahul.verma@example.com", "+91 98110 22222",
                 "45 Sector 18", "Noida", "Uttar Pradesh", "201301"),
    CustomerSeed("Ananya Iyer", "ananya.iyer@example.com", "+91 98400 33333",
                 "7 Besant Nagar 2nd Ave", "Chennai", "Tamil Nadu", "600090"),
    CustomerSeed("Arjun Mehta", "arjun.mehta@example.com", "+91 98200 44444",
                 "301 Hill Road, Bandra West", "Mumbai", "Maharashtra", "400050"),
    CustomerSeed("Sneha Das", "sneha.das@example.com", "+91 94350 55555",
                 "Zoo Narengi Road", "Guwahati", "Assam", "781024"),
    CustomerSeed("Vikram Singh", "vikram.singh@example.com", "+91 94140 66666",
                 "22 Civil Lines", "Jaipur", "Rajasthan", "302006"),
    CustomerSeed("Fatima Khan", "fatima.khan@example.com", "+91 98490 77777",
                 "Road No. 12, Banjara Hills", "Hyderabad", "Telangana", "500034"),
    CustomerSeed("Rohan Bose", "rohan.bose@example.com", "+91 94340 88888",
                 "Hakimpara", "Siliguri", "West Bengal", "734001"),
]


# --------------------------------------------------------------------------- #
# Insert helpers
# --------------------------------------------------------------------------- #
def insert_products(conn: sqlite3.Connection, created_at: str) -> dict[str, sqlite3.Row]:
    for p in PRODUCTS:
        conn.execute(
            """INSERT INTO products (sku, name, category, price_paise, stock_quantity,
                                     is_returnable, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (p.sku, p.name, p.category, p.price_paise, p.stock_quantity,
             int(p.is_returnable), created_at),
        )
    rows = conn.execute("SELECT * FROM products").fetchall()
    return {r["sku"]: r for r in rows}


def insert_customers(conn: sqlite3.Connection, created_at: str) -> list[int]:
    ids: list[int] = []
    for c in CUSTOMERS:
        cur = conn.execute(
            """INSERT INTO customers (name, email, phone, address_line, city, state,
                                      pincode, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (c.name, c.email, c.phone, c.address_line, c.city, c.state, c.pincode,
             created_at),
        )
        ids.append(int(cur.lastrowid))
    return ids


def create_order(
    conn: sqlite3.Connection,
    rng: random.Random,
    now: datetime,
    customer_id: int,
    address: str,
    items: list[tuple[sqlite3.Row, int]],
    status: OrderStatus,
    *,
    delivered_days_ago: int | None = None,
    placed_days_ago: int | None = None,
) -> tuple[int, list[int]]:
    """
    Insert an order with internally consistent totals and timestamps.

    For delivered orders pass `delivered_days_ago` — it is the start of the
    30-day return window, so curated scenarios pin it exactly. For other
    statuses pass `placed_days_ago`.

    Returns (order_id, [order_item_ids]).
    """
    # --- timeline ----------------------------------------------------------
    shipped_at = delivered_at = cancelled_at = None
    tracking: str | None = None

    if status == "delivered":
        assert delivered_days_ago is not None, "delivered orders need delivered_days_ago"
        delivered = now - timedelta(days=delivered_days_ago, hours=rng.randint(1, 8))
        shipped = delivered - timedelta(days=rng.randint(2, 4))
        placed = shipped - timedelta(days=rng.randint(1, 2))
        shipped_at, delivered_at = to_db_ts(shipped), to_db_ts(delivered)
    else:
        assert placed_days_ago is not None, f"{status} orders need placed_days_ago"
        placed = now - timedelta(days=placed_days_ago, hours=rng.randint(1, 8))
        if status == "shipped":
            shipped_at = to_db_ts(placed + timedelta(days=1, hours=rng.randint(1, 10)))
        elif status == "cancelled":
            cancelled_at = to_db_ts(placed + timedelta(hours=rng.randint(1, 12)))

    if status in ("shipped", "delivered"):
        tracking = f"TRK{rng.randint(10**9, 10**10 - 1)}"

    # --- money (integer paise only) -----------------------------------------
    subtotal = sum(p["price_paise"] * qty for p, qty in items)
    shipping = 0 if subtotal >= FREE_SHIPPING_MIN_PAISE else FLAT_SHIPPING_PAISE
    total = subtotal + shipping

    cur = conn.execute(
        """INSERT INTO orders (customer_id, status, subtotal_paise, shipping_paise,
                               total_paise, shipping_address, tracking_number,
                               placed_at, shipped_at, delivered_at, cancelled_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (customer_id, status, subtotal, shipping, total, address, tracking,
         to_db_ts(placed), shipped_at, delivered_at, cancelled_at),
    )
    order_id = int(cur.lastrowid)

    item_ids: list[int] = []
    for product, qty in items:
        # unit_price is snapshotted: later catalogue price changes must not
        # change what we refund for this order.
        icur = conn.execute(
            """INSERT INTO order_items (order_id, product_id, quantity, unit_price_paise)
               VALUES (?, ?, ?, ?)""",
            (order_id, product["id"], qty, product["price_paise"]),
        )
        item_ids.append(int(icur.lastrowid))
    return order_id, item_ids


def create_return(
    conn: sqlite3.Connection,
    now: datetime,
    *,
    order_id: int,
    order_item_id: int,
    customer_id: int,
    quantity: int,
    reason: str,
    status: str,
    requested_days_ago: int,
) -> int:
    unit_price = conn.execute(
        "SELECT unit_price_paise FROM order_items WHERE id = ?", (order_item_id,)
    ).fetchone()["unit_price_paise"]
    refund = unit_price * quantity
    cur = conn.execute(
        """INSERT INTO returns (order_id, order_item_id, customer_id, quantity, reason,
                                status, refund_amount_paise, requires_escalation,
                                requested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, order_item_id, customer_id, quantity, reason, status, refund,
         int(refund > REFUND_ESCALATION_THRESHOLD_PAISE),
         to_db_ts(now - timedelta(days=requested_days_ago))),
    )
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- #
# Main seeding routine
# --------------------------------------------------------------------------- #
def seed(db_path: str | Path | None = None, *, rng_seed: int = 42) -> dict[str, int]:
    rng = random.Random(rng_seed)
    now = utc_now()
    catalogue_created = to_db_ts(now - timedelta(days=180))

    init_db(db_path, reset=True)  # emails are UNIQUE, so reseeding needs a clean DB

    with get_db(db_path) as conn:
        p = insert_products(conn, catalogue_created)
        customer_ids = insert_customers(conn, to_db_ts(now - timedelta(days=120)))
        addr = {cid: c.shipping_address for cid, c in zip(customer_ids, CUSTOMERS)}

        # ---- Customer 1: Priya ------------------------------------------------
        priya = customer_ids[0]
        create_order(conn, rng, now, priya, addr[priya],                       # A
                     [(p["AP-TSH-001"], 2), (p["AC-WLT-001"], 1)], "delivered",
                     delivered_days_ago=5)
        create_order(conn, rng, now, priya, addr[priya],                       # B
                     [(p["FW-RUN-001"], 1)], "delivered", delivered_days_ago=10)
        create_order(conn, rng, now, priya, addr[priya],                       # C
                     [(p["AP-JNS-002"], 1)], "delivered", delivered_days_ago=45)
        create_order(conn, rng, now, priya, addr[priya],                       # D
                     [(p["EL-EAR-001"], 1)], "shipped", placed_days_ago=3)
        create_order(conn, rng, now, priya, addr[priya],                       # E
                     [(p["EL-PWR-002"], 1), (p["EL-CBL-003"], 2)], "processing",
                     placed_days_ago=0)
        create_order(conn, rng, now, priya, addr[priya],                       # F
                     [(p["AP-INN-004"], 1), (p["PC-SRM-001"], 1)], "delivered",
                     delivered_days_ago=3)
        create_order(conn, rng, now, priya, addr[priya],                       # G
                     [(p["AC-BAG-002"], 1)], "cancelled", placed_days_ago=20)

        # ---- Customer 2: Rahul ------------------------------------------------
        rahul = customer_ids[1]
        h_order, h_items = create_order(conn, rng, now, rahul, addr[rahul],    # H
                                        [(p["FW-SNK-003"], 1)], "delivered",
                                        delivered_days_ago=8)
        create_return(conn, now, order_id=h_order, order_item_id=h_items[0],
                      customer_id=rahul, quantity=1, reason="Size too small",
                      status="requested", requested_days_ago=1)

        i_order, i_items = create_order(conn, rng, now, rahul, addr[rahul],    # I
                                        [(p["AP-JKT-003"], 1)], "delivered",
                                        delivered_days_ago=15)
        i_return = create_return(conn, now, order_id=i_order, order_item_id=i_items[0],
                                 customer_id=rahul, quantity=1,
                                 reason="Zip broke after first use",
                                 status="requested", requested_days_ago=2)
        conn.execute(
            """INSERT INTO escalations (customer_id, order_id, return_id, reason,
                                        priority, status, created_at)
               VALUES (?, ?, ?, ?, 'normal', 'open', ?)""",
            (rahul, i_order, i_return,
             "Refund of Rs. 2,499.00 exceeds agent approval limit of Rs. 2,000.",
             to_db_ts(now - timedelta(days=2))),
        )

        # ---- Background customers: random orders ------------------------------
        all_products = list(p.values())
        for cid in customer_ids[2:]:
            for _ in range(rng.randint(2, 4)):
                picks = rng.sample(all_products, k=rng.randint(1, 3))
                items = [(prod, rng.randint(1, 2)) for prod in picks]
                status: OrderStatus = rng.choices(
                    ["delivered", "shipped", "processing", "cancelled"],
                    weights=[6, 2, 2, 1],
                )[0]
                if status == "delivered":
                    create_order(conn, rng, now, cid, addr[cid], items, status,
                                 delivered_days_ago=rng.randint(1, 60))
                else:
                    create_order(conn, rng, now, cid, addr[cid], items, status,
                                 placed_days_ago=rng.randint(0, 6))

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("customers", "products", "orders", "order_items",
                          "returns", "escalations")
        }
    return counts


def _print_demo_summary(db_path: str | Path | None) -> None:
    with get_db(db_path) as conn:
        rows = conn.execute(
            """SELECT o.id, c.name, o.status, o.total_paise,
                      CAST(julianday('now') - julianday(o.delivered_at) AS INTEGER) AS days
               FROM orders o JOIN customers c ON c.id = o.customer_id
               WHERE o.customer_id IN (1, 2) ORDER BY o.id"""
        ).fetchall()
    print("\nDemo orders (customers 1 & 2):")
    print(f"  {'id':>3}  {'customer':<13} {'status':<11} {'total':>14}  delivered")
    for r in rows:
        days = f"{r['days']}d ago" if r["days"] is not None else "-"
        print(f"  {r['id']:>3}  {r['name']:<13} {r['status']:<11} "
              f"{format_paise(r['total_paise']):>14}  {days}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wipe and seed the support database.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to the SQLite file")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for background data")
    args = parser.parse_args()

    result = seed(args.db, rng_seed=args.seed)
    print(f"Seeded {args.db}: " + ", ".join(f"{k}={v}" for k, v in result.items()))
    _print_demo_summary(args.db)
