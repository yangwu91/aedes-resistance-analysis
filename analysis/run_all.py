#!/usr/bin/env python3
"""
run_all.py -- Master pipeline script for the insecticide resistance meta-analysis.

Imports and runs each analysis script's main() function in order (a01 through a11).
Each step is wrapped in try/except so the pipeline continues even if one script fails.
Prints timing for each step and an overall summary at the end.
"""

import sys
import time
import importlib
from pathlib import Path

# Ensure the analysis directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPTS = [
    "a01_data_cleaning",
    "a02_descriptive_stats",
    "a03_meta_mortality",
    "a04_meta_rr",
    "a05_meta_kdr",
    "a06_meta_enzyme",
    "a07_cross_resistance",
    "a08_subgroup_analysis",
    "a09_meta_regression",
    "a10_publication_bias",
    "a11_sensitivity",
]


def main():
    print("=" * 70)
    print("INSECTICIDE RESISTANCE META-ANALYSIS -- MASTER PIPELINE")
    print("=" * 70)
    print(f"  Running {len(SCRIPTS)} analysis scripts in sequence.\n")

    overall_start = time.time()
    results = []

    for i, script_name in enumerate(SCRIPTS, start=1):
        step_label = f"[{i}/{len(SCRIPTS)}] {script_name}"
        print("\n" + "#" * 70)
        print(f"# {step_label}")
        print("#" * 70 + "\n")

        step_start = time.time()
        status = "SUCCESS"
        error_msg = ""

        try:
            module = importlib.import_module(script_name)
            if hasattr(module, "main"):
                module.main()
            else:
                status = "SKIPPED"
                error_msg = "No main() function found"
                print(f"  [WARNING] {script_name} has no main() function -- skipping.")
        except SystemExit as e:
            status = "FAILED"
            error_msg = f"SystemExit({e.code})"
            print(f"\n  [ERROR] {script_name} called sys.exit({e.code})")
        except Exception as e:
            status = "FAILED"
            error_msg = str(e)
            print(f"\n  [ERROR] {script_name} failed with exception:")
            print(f"    {type(e).__name__}: {e}")

        step_elapsed = time.time() - step_start
        results.append({
            "step": i,
            "script": script_name,
            "status": status,
            "time_s": round(step_elapsed, 2),
            "error": error_msg,
        })

        status_symbol = {"SUCCESS": "OK", "FAILED": "FAIL", "SKIPPED": "SKIP"}[status]
        print(f"\n  [{status_symbol}] {step_label} -- {step_elapsed:.1f}s")

    overall_elapsed = time.time() - overall_start

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    n_success = sum(1 for r in results if r["status"] == "SUCCESS")
    n_failed = sum(1 for r in results if r["status"] == "FAILED")
    n_skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    print(f"\n  {'Step':>4s}  {'Script':<30s}  {'Status':>8s}  {'Time (s)':>10s}")
    print(f"  {'----':>4s}  {'-'*30:<30s}  {'--------':>8s}  {'-'*10:>10s}")

    for r in results:
        print(
            f"  {r['step']:4d}  {r['script']:<30s}  {r['status']:>8s}  {r['time_s']:10.2f}"
        )
        if r["error"]:
            print(f"        Error: {r['error']}")

    print(f"\n  Total: {n_success} succeeded, {n_failed} failed, {n_skipped} skipped")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} min)")

    if n_failed > 0:
        print(f"\n  WARNING: {n_failed} script(s) failed. Check errors above.")
        failed_scripts = [r["script"] for r in results if r["status"] == "FAILED"]
        for s in failed_scripts:
            print(f"    - {s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
