# AmazonHelp support agent

An AI customer-support agent for **@AmazonHelp**, built from the Kaggle
*Customer Support on Twitter* dataset (2.8M tweets, 108 brands). For every
incoming customer tweet it makes three decisions:

1. **Classify** it into one of eight intents defined from the data.
2. **Draft a reply** grounded in how AmazonHelp actually answered similar messages.
3. **Route** it — handle automatically, or escalate to a human, with a stated reason.

| | intent macro-F1 | routing accuracy | missed escalations | reply sendable (blind human) |
|---|---|---|---|---|
| trivial baseline | 0.035 | 0.700 | 66 / 66 (100%) | 48% |
| simple baseline | 0.541 | 0.823 | 35 / 66 (53%) | 52% |
| **agent** | **0.797** | **0.873** | **24 / 66 (36%)** | **96%** |

> 📄 **[REPORT.md](REPORT.md)** — the full write-up: problem framing, results
> against both baselines, top 5 failure modes, and the mandatory
> *"what is misleading about my headline number"* section.
>
> 📝 **[DECISIONS.md](DECISIONS.md)** — 27 non-obvious decisions and why.

---

## Contents

- [Quick start](#quick-start) — install, then three ways to run it
- [How it works](#how-it-works) — the pipeline, stage by stage
- [The eight intents](#the-eight-intents)
- [The escalation policy](#the-escalation-policy)
- [The golden evaluation set](#the-golden-evaluation-set)
- [How it is evaluated](#how-it-is-evaluated)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick start

### 1. Install

Python 3.10+ and two packages:

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download `twcs.csv` from
[Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(516 MB) into the repo root, then build the conversation index once:

```bash
python src/prepare_data.py
```

*~4 minutes. Two streamed passes over 2.8M tweets →
`data/amazonhelp_threads.jsonl` (81,759 conversations). You only do this once.*

### 3. Add an LLM key

Copy `.env.example` to `.env` in the repo root and fill in one provider:

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```

*Only the agent and the judge need a key. Both baselines run without one.*

---

### ▶ Option A — talk to the agent (chat UI)

The fastest way to see whether it works.

```bash
python src/webapp.py
```

Open **http://localhost:8000**. A chat interface: type a customer message, get
the brand's drafted reply. Under each reply a badge says whether it was **sent
automatically** or **held for a human**, with the reason. A *details* toggle
reveals the intent, which rules fired, and the past AmazonHelp conversations the
draft was grounded on.

*~1 minute to load the index at startup, then a few seconds per reply.
`Ctrl+C` to stop.*

Try these — each takes a different path through the routing policy:

| message | expected |
|---|---|
| *my parcel was supposed to arrive yesterday* | ✅ sent automatically |
| *this is the third time and no one has replied* | ⚠️ held — `repeat_contact` |
| *there are unauthorised transactions on my card* | ⚠️ held — `security_or_fraud` |
| *i can't log into my account* | ⚠️ held — `account_security_policy` |
| *mi pedido no ha llegado* | ⚠️ held — `non_english` |
| *just got my order in 6 hours, incredible* | ✅ sent automatically |

> **Note:** the UI looks like a chatbot, but every message is handled
> independently as a fresh incoming tweet. The agent is opening-message triage
> and carries no conversation state — a follow-up like *"what about my other
> order?"* will not remember the previous turn. A deliberate scope decision, not
> a bug ([REPORT.md §2](REPORT.md#2-problem-framing)).

### ▶ Option B — reproduce the results

```bash
python run_all.py --skip-prep                # baselines only, no API key  (~2 min)
python run_all.py --skip-prep --with-agent   # + agent, judge, kappa       (~25 min)
```

Prints metrics at every stage and writes everything to `results/`.

### ▶ Option C — the terminal

```bash
python src/try_agent.py                  # interactive REPL
python src/try_agent.py --demo           # 8 probe messages, one per routing branch
python src/agent.py "your message here"  # one message, full JSON output
python src/retrieval.py "late parcel"    # retrieval only, to inspect precedent
python src/intents.py                    # print the taxonomy and rules
python src/evaluate.py --system agent    # score one system on the golden set
python src/failures.py --system agent    # failure analysis
python src/judge.py --agreement          # judge vs human, Cohen's kappa
```

---

## How it works

```
twcs.csv  (2.8M tweets, 108 brands)
    │
    │  ① prepare_data.py — reconstruct conversations
    ▼
81,759 AmazonHelp conversations
    │
    │  ② dataset.py — language gate + split
    ▼
┌────────────────────────────┬─────────────────────────┐
│  retrieval corpus  45,073  │  held-out eval pool     │
│  (the agent's "memory")    │       11,389            │
└────────────────────────────┴─────────────────────────┘
    │                                   │
    │  ③ retrieval.py                   │  ④ build_golden.py
    ▼                                   ▼
TF-IDF precedent index          220 hand-labelled examples
    │                                   │
    └─────────► ⑤ agent.py ◄────────────┘
                    │
                    │  ⑥ evaluate.py · judge.py · failures.py
                    ▼
                results/
```

### ① Conversation reconstruction — `prepare_data.py`

Two streamed passes over the CSV (it is never loaded into memory):

- **Pass 1** builds a union-find over `in_response_to_tweet_id` for *all* 2.8M
  tweets, and records which tweets AmazonHelp wrote.
- **Pass 2** re-reads and keeps full text only for connected components
  AmazonHelp took part in, orders each thread by hops-to-root, and emits
  *(customer's opening message, brand's first reply)*.

**Why a graph rather than a filter?** Customers often open with *"Amazon is
useless"* and **no @handle** — the brand picks the complaint up itself. Filtering
on the handle decapitates those threads. My first version did exactly that, and
only **43%** of the extracted "opening messages" were real openers; the rest were
mid-thread fragments. The graph approach gets **98.7%**.

### ② Language gate and split — `dataset.py`

```
81,759 conversations
  → thread-openers only        80,657
  → English (heuristic)        56,462     ← 41% is not English, routed to humans
  → split on MD5(conversation_id):
        80% → retrieval corpus 45,073
        20% → held-out pool    11,389
```

The split prevents leakage: the golden set is drawn only from the held-out pool,
so **no evaluated message can retrieve its own historical reply as precedent**.
MD5 rather than Python's `hash()`, which is salted per process and would
reshuffle the split between runs.

### ③ Retrieval — `retrieval.py`

TF-IDF over **word 1–2 grams + character 3–5 grams**. The char n-grams matter:
this text is dense with typos and they absorb them. Order and tracking numbers
collapse to a `<num>` token, since they are unique per customer and only dilute
the vocabulary.

Given a new message it returns the *k* most similar past messages **and the
brand's actual replies to them**. That is what "grounded in how the brand has
historically resolved similar issues" means concretely.

### ⑤ The agent — `agent.py`

```
incoming message
   → retrieve 5 similar past conversations
   → ONE LLM call → intent + confidence + drafted reply + needs_human
   → hard regex rules run independently
   → decide_route():
         not English?          → escalate  (non_english)
         hard rule fired?      → escalate  (cannot be overridden)
         intent = account?     → escalate  (account_security_policy)
         confidence < 0.55?    → escalate  (low_confidence)
         model raised a flag?  → escalate  (model_flagged)
         otherwise             → auto
```

Classification and drafting are **one** LLM call, not two — the retrieved
precedents are the expensive part of the prompt and both steps need them.

**Routing is rule-first, and the LLM may only *add* escalations, never remove
them.** The errors are not symmetric: over-escalating a routine question costs an
agent thirty seconds; auto-replying to a fraud report does not. A model that can
talk itself out of escalating is strictly worse than a regex that cannot.

The agent writes the literal placeholder `<link>` and is forbidden from inventing
URLs; the chat UI resolves it to a real Amazon help URL at render time. A model
generating URLs freely produces plausible-looking wrong ones.

---

## The eight intents

Derived from term frequencies over 80k messages, **not** declared up front: seven
recurring actionable clusters, plus a genuine eighth for the ~21% matching none.

| intent | covers |
|---|---|
| `delivery_issue` | shipment in motion gone wrong — late, lost, marked delivered but not received, damaged in transit, carrier conduct |
| `order_status` | pre-delivery questions — has it shipped, can I cancel, pre-order timing, duplicate item sent |
| `refund_return` | returns, replacements, exchanges, warranty claims, refunds that have not appeared |
| `payment_billing` | charged twice, card declined, gift card / promo / cashback, price mismatch, invoices |
| `prime_membership` | the subscription itself, including unwanted or post-cancellation membership charges |
| `account_access` | login, password, 2FA, locked or compromised account, account settings |
| `product_or_app_issue` | item defective / wrong / counterfeit, or app, website, Prime Video, Kindle, Alexa, Fire failures |
| `feedback_or_other` | no actionable request — praise, rants, jokes, catalogue and feature requests, seller enquiries |

**35% of messages match more than one intent's keywords**, so single-label
classification needs a written tie-break or it is unmeasurable — two reasonable
people would disagree and it would look like model error. The priority order,
ranked by what the brand must act on first:

```
account_access > delivery_issue > refund_return > payment_billing >
prime_membership > product_or_app_issue > order_status > feedback_or_other
```

Four further tie-breaks were written down *during* labelling, when I found I was
deciding them inconsistently. They live in `src/intents.py` and at the top of
`src/_golden_labels.py`.

## The escalation policy

Eight hard rules fire **before** the model is consulted and cannot be overridden:

| rule | fires on |
|---|---|
| `legal_or_media` | lawyer, lawsuit, ombudsman, journalist, consumer court |
| `security_or_fraud` | hacked, fraud, stolen, unauthorised, phishing |
| `safety_or_harm` | injury, burn, fire hazard, poison, hospital, allergy |
| `vulnerable_or_distress` | disability, elderly, bereavement, cannot afford |
| `repeat_contact` | ≥3 stated occurrences, or already contacted support unresolved |
| `pii_in_public` | phone number, email address or login id in a public tweet |
| `business_or_seller_dispute` | seller account, brand listing, affiliate, vendor |
| `abuse_or_severe_anger` | profanity, "worst company", threats to cancel |

Plus: **`account_access` always escalates by policy** — the standard reply for
account problems would invite credentials into a public thread. This deliberately
over-escalates benign account questions; that cost is accepted, and visible in
the failure analysis.

## The golden evaluation set

**220 examples, hand-labelled**, in `data/golden.jsonl` — frozen and committed.

**Sampled** only from the held-out pool. A uniform random 220 would contain ~12
`account_access` examples and one or two of most escalation triggers, too few to
measure per-class — so it is stratified with a quota per intent, a quota of
messages a keyword prior *cannot* place, and a quota that trips an escalation
rule.

**Labelled** by reading each message against written definitions. The keyword
prior chose only *what to read* and was never shown while labelling; the sample
was shuffled so labelling order did not follow the strata.

| intent | n | | route | n |
|---|---|---|---|---|
| `product_or_app_issue` | 47 | | auto | 154 (70%) |
| `feedback_or_other` | 45 | | escalate | 66 (30%) |
| `delivery_issue` | 36 | | | |
| `refund_return` | 25 | | **top escalation reasons** | |
| `payment_billing` | 24 | | `repeat_contact` | 31 |
| `order_status` | 24 | | `security_or_fraud` | 11 |
| `account_access` | 10 | | `abuse_or_severe_anger` | 10 |
| `prime_membership` | 9 | | `business_or_seller_dispute` | 5 |

> ⚠️ Labels are keyed by **conversation id**, and `data/golden_unlabelled.jsonl`
> is a **frozen artifact**. `build_golden.py` documents how the sample was drawn,
> but re-running it is *not* part of reproducing results — it would draw a
> different sample that has no labels. `finalise_golden.py` asserts on any
> mismatch. [DECISIONS.md #27](DECISIONS.md) explains why this is a hard rule.

## How it is evaluated

Three independent measurements, because no single one is trustworthy.

**1 — Automated** (`evaluate.py`): intent accuracy and macro-F1, plus routing
with **missed and over-escalations reported separately** — collapsing them into
one accuracy number hides that the two errors have very different costs. Reply
checks are deterministic regexes for invented URLs, fabricated facts and promised
timeframes: they measure *did not fabricate*, nothing more.

**2 — LLM-as-judge** (`judge.py`): a four-dimension rubric plus a binary
*send without review* verdict.

**3 — Blind human rating** (`_judge_human.py`): 75 replies, 25 per system,
shuffled with the producing system hidden, rated before the judge ran.

That third check is what caught the judge **ranking the trivial baseline
highest** (Cohen's kappa 0.218). The reply-quality claim in the report therefore
rests on the human ratings, not the judge — see
[REPORT.md §4](REPORT.md#4-results-against-two-baselines).

---

## Repository layout

| file | what it does |
|---|---|
| **`REPORT.md`** | the full write-up — framing, results, failures, misleading section |
| **`DECISIONS.md`** | 27 non-obvious decisions and why |
| `run_all.py` | runs the whole pipeline end to end |
| `src/prepare_data.py` | `twcs.csv` → 81,759 reconstructed conversations |
| `src/dataset.py` | language gate, corpus / eval split |
| **`src/intents.py`** | the taxonomy and escalation policy — **read this one first** |
| `src/retrieval.py` | TF-IDF precedent index over 45k historical replies |
| `src/agent.py` | classify → draft → route |
| `src/baselines.py` | the trivial and simple baselines |
| `src/evaluate.py` | automated metrics |
| `src/judge.py` | LLM-as-judge and the kappa against human ratings |
| `src/failures.py` | failure analysis |
| `src/webapp.py` | the chat UI |
| `src/try_agent.py` | terminal tester |
| `src/llm.py` | provider-agnostic LLM client with disk cache |
| `src/build_golden.py` | how the 220 were sampled (provenance; not re-run) |
| `src/_golden_labels.py` | the 220 hand-assigned labels, keyed by conversation id |
| `src/_judge_human.py` | 75 blind human ratings the judge is validated against |
| `data/golden.jsonl` | **the golden set** — frozen, committed |

**Dependencies:** `scikit-learn` and `requests`. Everything else is the standard
library — including the web UI, which is `http.server`. Deliberate: the project
should run anywhere without a heavyweight environment.

## Configuration

`.env` in the repo root (gitignored — never commit a key):

```ini
LLM_PROVIDER=openrouter          # openrouter | groq | nvidia
OPENROUTER_API_KEY=sk-or-...
# LLM_MODEL=meta-llama/llama-3.3-70b-instruct    # optional override
```

All three providers speak the same OpenAI-compatible API, so one client
(`src/llm.py`) covers them. Reported results use **OpenRouter + Llama 3.3 70B
Instruct**.

Every call runs at **temperature 0** and is **cached on disk** under
`data/.llm_cache/`, so re-runs are instant and byte-identical. An evaluation
whose numbers move between runs cannot be used to compare two systems.

> **On the 15-minute budget.** The no-key path — data prep, golden set, both
> baselines — is ~6 minutes including the CSV scan, and reproduces both required
> baselines. `--with-agent` is bound by provider rate limits, not compute: 220
> agent calls plus 440 judge calls. Measured **~25 min on OpenRouter**; over an
> hour on Groq's free tier (8,000 tokens/minute cap). Use `--limit 60` for a full
> run inside 15 minutes. All reported results are from the full 220.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `no LLM configured` | No `.env`, or the key name does not match `LLM_PROVIDER`. Copy `.env.example`. |
| `LLM_PROVIDER appears more than once` | Duplicated key in `.env`; the first wins. Delete the extra line. |
| `twcs.csv not found` | Download from Kaggle into the repo root, then run `python src/prepare_data.py`. |
| Agent run stalls with `429` | Provider rate limit. Groq's free tier is 8k tokens/min — switch to OpenRouter or use `--limit`. |
| Empty replies on Groq | `gpt-oss-120b` is a reasoning model whose chain of thought consumes `max_tokens`. Handled by `reasoning_effort: "low"` + retry; see [DECISIONS.md #21](DECISIONS.md). |
| `UnicodeEncodeError` on Windows | Every entry point calls `console.init()` to force UTF-8. If you add a script, call it too. |
| Numbers differ from the report | You re-ran `build_golden.py`. The golden set is frozen — restore it and run `finalise_golden.py`. |

---

*Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(Kaggle, `thoughtvector`). Cohen's kappa is implemented directly in `src/judge.py`
so the agreement number is auditable. Banking77 was not used. AI coding assistance
was used throughout, per the assignment's rules.*
