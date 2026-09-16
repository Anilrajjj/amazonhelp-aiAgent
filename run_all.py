"""Reproduce the headline results end to end.

    python run_all.py                 # data prep + golden set + both baselines
    python run_all.py --with-agent    # also the LLM agent and the judge

Everything except --with-agent runs with no API key.
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def step(title, *args):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable] + list(args), cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"step failed: {title}")
    print(f"[{time.time() - t:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-agent", action="store_true")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse data/amazonhelp_threads.jsonl if already built")
    a = ap.parse_args()

    threads = os.path.join(ROOT, "data", "amazonhelp_threads.jsonl")
    if not (a.skip_prep and os.path.exists(threads)):
        step("1. extract AmazonHelp conversations from twcs.csv", "prepare_data.py")
    # NOT build_golden.py: data/golden_unlabelled.jsonl is a frozen artifact of
    # the exact 220 messages that were hand-labelled. Re-sampling would draw a
    # different set that has no labels, so build_golden.py records provenance
    # only and is run manually.
    step("2. join hand labels -> data/golden.jsonl", "finalise_golden.py")
    step("3. baselines", "evaluate.py", "--system", "trivial")
    step("4. baselines", "evaluate.py", "--system", "simple")
    if a.with_agent:
        step("5. agent", "evaluate.py", "--system", "agent")
        step("6. judge: agent", "judge.py", "--run", "agent")
        step("7. judge: simple", "judge.py", "--run", "simple")
        step("8. sample replies for human rating", "judge.py", "--sample")
        step("9. judge vs human agreement (Cohen's kappa)", "judge.py", "--agreement")
    print("\nDone. Metrics in results/metrics.json")


if __name__ == "__main__":
    main()
