"""Loading, language filtering and the corpus/eval split.

Everything downstream reads the data through here so that one rule decides what
counts as an in-scope message, and so the retrieval corpus and the evaluation
pool can never overlap.
"""
import hashlib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREADS = os.path.join(ROOT, "data", "amazonhelp_threads.jsonl")

# Function words that are common in English and rare in the other languages in
# this dump (Spanish, Portuguese, Italian, French, Japanese, Arabic). Cheap and
# dependency-free; we only need to separate "English" from "not English", and
# the borderline cases get escalated anyway.
_EN_MARKERS = set("""the and you your for that have with this not was are but from they what
when will please can it's i'm my me is has been did does why how still get got
order delivery item now about there their would could should""".split())


def ascii_ratio(s):
    return sum(1 for c in s if ord(c) < 128) / max(len(s), 1)


def is_english(text):
    """Heuristic language gate. Non-English messages are out of scope for the
    agent (see README: what we chose not to build) and are routed to a human."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return ascii_ratio(text) > 0.9 and len(words & _EN_MARKERS) >= 2


def _bucket(conversation_id, n=100):
    """Stable hash bucket 0..n-1 -- deterministic across runs and machines,
    unlike hash(), which is salted per process."""
    h = hashlib.md5(str(conversation_id).encode()).hexdigest()
    return int(h[:8], 16) % n


def load_all(path=THREADS):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run: python src/prepare_data.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(path=THREADS, eval_pct=20):
    """Return (corpus, eval_pool).

    corpus    -- English openers that have a brand reply; the agent retrieves
                 precedent from these.
    eval_pool -- English openers held out entirely; the golden set is sampled
                 from here, so no evaluated message can retrieve its own reply.
    """
    recs = [r for r in load_all(path)
            if r["opens_thread"] and is_english(r["customer_message"])]
    corpus, pool = [], []
    for r in recs:
        (pool if _bucket(r["conversation_id"]) < eval_pct else corpus).append(r)
    corpus = [r for r in corpus if r.get("brand_reply")]
    return corpus, pool


if __name__ == "__main__":
    import console; console.init()
    all_recs = load_all()
    corpus, pool = load_split()
    print(f"conversations           {len(all_recs):,}")
    print(f"  thread openers        {sum(1 for r in all_recs if r['opens_thread']):,}")
    print(f"  english openers       {len(corpus) + len(pool):,}")
    print(f"retrieval corpus (80%)  {len(corpus):,}  (english openers with a brand reply)")
    print(f"eval pool        (20%)  {len(pool):,}")
