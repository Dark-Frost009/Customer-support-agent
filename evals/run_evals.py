"""
Run the eval suite against the real OpenAI model.

    python evals/run_evals.py                  # all cases, once
    python evals/run_evals.py --repeat 3       # each case 3x (models are non-deterministic)
    python evals/run_evals.py -k refund -v     # only cases matching "refund", show transcripts
    python evals/run_evals.py --model gpt-4o   # compare models
    python evals/run_evals.py --list

Needs OPENAI_API_KEY. Uses a temporary database and policy index, so your
real support.db is never touched. A full run with gpt-4o-mini costs a few
rupees. Results are saved to evals/results/ as JSON.

Exit code is 1 if the pass rate is below --threshold, so this can gate a
release in CI (with an API key secret) once you're happy with the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Point the app at throwaway files BEFORE importing it: db.py and rag.py read
# these environment variables at import time.
_TMP = Path(tempfile.mkdtemp(prefix="support_agent_evals_"))
os.environ["SUPPORT_DB_PATH"] = str(_TMP / "eval_support.db")
os.environ["POLICY_INDEX_PATH"] = str(_TMP / "eval_policy_index.db")
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "evals")]

import agent  # noqa: E402
import rag  # noqa: E402
from cases import CASES  # noqa: E402
from harness import CaseResult, run_case  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-model evals for the support agent.")
    parser.add_argument("-k", "--filter", default="", help="only run cases whose id contains this")
    parser.add_argument("--model", default=agent.MODEL, help=f"chat model (default {agent.MODEL})")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--threshold", type=float, default=0.9, help="minimum pass rate, 0-1")
    parser.add_argument("-v", "--verbose", action="store_true", help="print transcripts")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    cases = [c for c in CASES if args.filter in c.id]
    if args.list:
        for c in cases:
            print(f"{c.id:<36} {c.description}")
        return 0
    if not cases:
        print(f"No cases match {args.filter!r}. Use --list to see them.")
        return 2
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Evals call the real model.")
        return 2

    print("Building policy index (one embeddings call)...")
    rag.build_index(force=True)
    client = agent.default_client()

    results: list[CaseResult] = []
    total_runs = len(cases) * args.repeat
    print(f"Running {len(cases)} case(s) x {args.repeat} on {args.model}\n")
    for case in cases:
        for i in range(args.repeat):
            r = run_case(case, client, args.model)
            results.append(r)
            label = f"{case.id}" + (f" [{i + 1}/{args.repeat}]" if args.repeat > 1 else "")
            print(f"  {'PASS' if r.passed else 'FAIL'}  {label:<44} {r.seconds:5.1f}s")
            for f in r.failures:
                print(f"        - {f}")
            if args.verbose or not r.passed:
                for t in r.transcript:
                    print(f"        you:   {t['user']}")
                    print(f"        agent: {t['agent'][:300]}" + ("..." if len(t['agent']) > 300 else ""))
                    print(f"               tools: {', '.join(t['tools']) or '-'}")

    passed = sum(r.passed for r in results)
    rate = passed / total_runs
    print(f"\n{passed}/{total_runs} passed ({rate:.0%}) on {args.model}")

    # Per-case pass rate when repeating, to spot flaky behaviour.
    if args.repeat > 1:
        print("\nPer case:")
        for case in cases:
            rs = [r for r in results if r.case_id == case.id]
            print(f"  {sum(r.passed for r in rs)}/{len(rs)}  {case.id}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-{args.model}.json"
    out.write_text(json.dumps({
        "model": args.model,
        "repeat": args.repeat,
        "passed": passed,
        "total": total_runs,
        "pass_rate": rate,
        "results": [r.__dict__ for r in results],
    }, indent=2, default=str), encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")

    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
