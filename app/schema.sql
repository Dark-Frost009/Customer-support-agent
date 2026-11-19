-- Customer support agent schema (SQLite)
-- All money values are INTEGER paise (1 INR = 100 paise). Never store floats.
-- Timestamps are ISO-8601 UTC text, e.g. '2026-11-19 05:32:00'.

PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    email        TEXT    NOT NULL UNIQUE,
    phone        TEXT,
    address_line TEXT,
    city         TEXT,
    state        TEXT,
    pincode      TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE products (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    sku            TEXT    NOT NULL UNIQUE,
    name           TEXT    NOT NULL,
    category       TEXT    NOT NULL,            -- e.g. 'footwear', 'apparel', 'accessories', 'electronics'
    price_paise    INTEGER NOT NULL CHECK (price_paise >= 0),
    stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    is_returnable  INTEGER NOT NULL DEFAULT 1 CHECK (is_returnable IN (0, 1)),
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      INTEGER NOT NULL REFERENCES customers(id),
    status           TEXT    NOT NULL DEFAULT 'processing'
                     CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled')),
    subtotal_paise   INTEGER NOT NULL CHECK (subtotal_paise >= 0),
    shipping_paise   INTEGER NOT NULL DEFAULT 0 CHECK (shipping_paise >= 0),
    total_paise      INTEGER NOT NULL CHECK (total_paise >= 0),
    shipping_address TEXT    NOT NULL,
    tracking_number  TEXT,
    placed_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    shipped_at       TEXT,
    delivered_at     TEXT,                      -- start of the 30-day return window
    cancelled_at     TEXT
);

CREATE TABLE order_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id          INTEGER NOT NULL REFERENCES orders(id),
    product_id        INTEGER NOT NULL REFERENCES products(id),
    quantity          INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_paise  INTEGER NOT NULL CHECK (unit_price_paise >= 0)  -- price at time of purchase
);

CREATE TABLE returns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id              INTEGER NOT NULL REFERENCES orders(id),
    order_item_id         INTEGER NOT NULL REFERENCES order_items(id),
    customer_id           INTEGER NOT NULL REFERENCES customers(id),
    quantity              INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    reason                TEXT    NOT NULL,
    status                TEXT    NOT NULL DEFAULT 'requested'
                          CHECK (status IN ('requested', 'approved', 'rejected', 'received', 'refunded')),
    refund_amount_paise   INTEGER NOT NULL CHECK (refund_amount_paise >= 0),
    requires_escalation   INTEGER NOT NULL DEFAULT 0 CHECK (requires_escalation IN (0, 1)),
                          -- set to 1 when refund_amount_paise > 200000 (INR 2,000)
    requested_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    resolved_at           TEXT
);

CREATE TABLE escalations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      INTEGER NOT NULL REFERENCES customers(id),
    order_id         INTEGER REFERENCES orders(id),
    return_id        INTEGER REFERENCES returns(id),
    reason           TEXT    NOT NULL,
    priority         TEXT    NOT NULL DEFAULT 'normal'
                     CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    status           TEXT    NOT NULL DEFAULT 'open'
                     CHECK (status IN ('open', 'in_progress', 'resolved')),
    assigned_to      TEXT,
    resolution_notes TEXT,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    resolved_at      TEXT
);

CREATE INDEX idx_orders_customer      ON orders(customer_id);
CREATE INDEX idx_orders_status        ON orders(status);
CREATE INDEX idx_order_items_order    ON order_items(order_id);
CREATE INDEX idx_returns_order        ON returns(order_id);
CREATE INDEX idx_returns_customer     ON returns(customer_id);
CREATE INDEX idx_escalations_customer ON escalations(customer_id);
CREATE INDEX idx_escalations_status   ON escalations(status);
