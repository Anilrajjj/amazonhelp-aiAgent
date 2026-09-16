# An AI support agent for @AmazonHelp

**Report — Hiver SDE Intern take-home**

A triage agent for Amazon's Twitter support account. For every incoming customer
tweet it classifies the intent, drafts a reply grounded in how the brand has
historically answered similar messages, and decides whether the reply can go out
automatically or must be held for a human.

Everything below is measured on a **220-example golden set I labelled by hand**.
Reproduction instructions are in [README.md](README.md); the reasoning behind
individual choices is in [DECISIONS.md](DECISIONS.md).

---

## 1. Headline results

| | intent macro-F1 | routing accuracy | **missed escalations** | reply sendable (blind human) |
|---|---|---|---|---|
| trivial baseline | 0.035 | 0.700 | 66 / 66 (100%) | 48% |
| simple baseline | 0.541 | 0.823 | 35 / 66 (53%) | 52% |
| **agent** | **0.797** | **0.873** | **24 / 66 (36%)** | **96%** |

The agent roughly doubles the simple baseline's macro-F1 and cuts missed
escalations by a third, while holding over-escalation flat at 2.6%. It is the
only system with zero reply-safety violations.

**Section 5 explains why you should not trust this table on its own.**

---

## 2. Problem framing

### What "good" means for this brand

I read several hundred of AmazonHelp's real replies before defining anything.
The pattern is near-invariant: acknowledge the specific problem, apologise, move
the customer to a private channel. The brand **cannot see orders or accounts from
Twitter**, so it cannot resolve a case in public even in principle.

That rules out the obvious success metric. "Did the agent solve the customer's
problem?" would score near zero for a system behaving exactly like the brand
does. So:

> **Auto-handled means "this reply is safe to post without a human reading it
> first" — not "this case is closed."**

Under that definition, good means three things:

1. **The intent is right**, because intent decides which queue and which macro a
   human eventually picks up.
2. **The reply invents nothing.** Stating a delivery date, refund amount or order
   status the brand cannot know is the fastest way to make a support account
   worse. A vague-but-true reply beats a specific-but-fabricated one.
3. **The genuinely dangerous messages reach a human** — fraud, legal threats,
   safety issues, vulnerable customers, and people already failed several times.

**Point 3 dominates, because the two routing errors are not symmetric.**
Over-escalating a routine question costs an agent thirty seconds. Auto-replying
to a fraud report with a cheerful template is the kind of thing that gets
screenshotted. The evaluation therefore reports **missed escalations separately**
and treats that, not accuracy, as the number that decides deployment.

### What I chose not to build

- **Non-English support.** 41% of AmazonHelp's thread-openers are not English
  (Spanish, Portuguese, Japanese, Italian, Arabic). Doing it properly needs
  per-language retrieval and per-language quality evaluation, which I could not
  evaluate honestly in the time. Non-English is detected and routed to a human.
  The detector is a crude heuristic and it leaks — two leaks are in the golden
  set, labelled as the failures they are.
- **Multi-turn conversation.** The agent acts on the opening message only. All
  three decisions are made on arrival, before any back-and-forth exists.
- **Actually resolving anything.** No order lookup, no refunds, no account
  actions. There is no system to call, and pretending otherwise would make the
  numbers meaningless.
- **Fine-tuning.** With 220 labelled examples the honest use is evaluation, not
  training. A train split would have left ~60 for testing, which measures nothing.
- **Embedding retrieval.** TF-IDF with character n-grams handles typo-dense short
  text well and keeps the pipeline reproducible on a laptop with no model
  download. First thing I would upgrade, not a limitation I am defending.

---

## 3. Approach

### Data: the fix that mattered most

| | |
|---|---|
| Raw tweets | 2,811,774 across 108 brands |
| AmazonHelp outbound | 169,840 — largest in the dump (AppleSupport: 106,860) |
| Conversations reconstructed | **81,759** |
| …starting at the true head of their thread | 80,657 (**98.7%**) |
| …in English | 56,462 (59%) |
| Retrieval corpus / held-out eval pool | 45,073 / 11,389 |

My first extraction kept rows where AmazonHelp was the author or the text
contained `@amazonhelp`. It produced a plausible 54k conversations — of which
**only 43% actually started a conversation.** Customers routinely open with
*"Amazon is useless"* and no handle, and the brand picks the complaint up itself;
handle-filtering decapitates those threads, leaving mid-thread fragments like
*"That would be AMZL US. didn't even stop."* being classified as opening messages.

The fix: build a union-find over `in_response_to_tweet_id` across all 2.8M
tweets, keep whole connected components AmazonHelp took part in, *then* filter.
True-opener rate went **43% → 98.7%**. This was the highest-leverage change in
the project and it was invisible in every summary statistic.

**20% of conversations are held out before anything else touches them**, and the
golden set is drawn only from that pool — so no evaluated message can retrieve
its own historical reply as precedent.

### Intents

Eight labels derived from term frequencies over 80k messages, not declared up
front: seven recurring actionable clusters, plus `feedback_or_other` for the ~21%
matching none of them (praise, rants, jokes, catalogue requests) — a genuine
class, not a bin.

`delivery_issue` · `order_status` · `refund_return` · `payment_billing` ·
`prime_membership` · `account_access` · `product_or_app_issue` · `feedback_or_other`

**35% of messages match more than one intent's keywords**, so single-label
classification needs a written tie-break or it is unmeasurable — two reasonable
people would disagree and it would look like model error. The priority order runs
`account_access > delivery_issue > refund_return > payment_billing >
prime_membership > product_or_app_issue > order_status > feedback_or_other`,
ranked by what the brand must act on first.

### The agent

```
incoming message
   → retrieve 5 most similar past conversations (TF-IDF, word + char n-grams)
   → ONE LLM call: intent + confidence + drafted reply + needs_human
   → hard regex rules run independently
   → decide_route():  non-English      → escalate
                      hard rule fired  → escalate   (cannot be overridden)
                      account_access   → escalate   (policy)
                      confidence < 0.55 → escalate
                      model flagged     → escalate
                      otherwise         → auto
```

**The LLM may only *add* escalations, never remove them.** Eight hard rules —
legal, fraud, safety, vulnerability, repeat contact, PII, seller dispute, abuse —
fire before the model is consulted. A model that can talk itself out of
escalating a fraud report is strictly worse than a regex that cannot.

### Golden evaluation set — 220 examples

**Sampling.** A uniform random 220 would contain ~12 `account_access` examples
and one or two of most escalation triggers, too few to measure per-class. So it
is stratified: a quota per intent from a cheap keyword prior, a quota of messages
the prior cannot place at all, and a quota that trips an escalation rule.

**Labelling.** The prior chose only *what to read*; it was never shown during
labelling, and the sample was shuffled so labelling order did not follow the
strata. Every label was assigned by reading the message against written
definitions. Where I caught myself deciding the same ambiguity inconsistently, I
stopped, wrote the rule down, and applied it uniformly — those four tie-breaks
are in `src/_golden_labels.py`.

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

---

## 4. Results against two baselines

**trivial** — predicts the majority intent for everything, never escalates, sends
one canned reply. It exists to show what the numbers look like when the system
does nothing.

**simple** — no LLM anywhere. TF-IDF + logistic regression on weak keyword labels
for intent; the nearest historical AmazonHelp reply **copied verbatim** as the
reply; hard rules only for routing. This is deliberately a strong baseline: the
brand's replies are templated, so copy-the-nearest-neighbour is competitive, and
if generation cannot beat it then generation is not earning its cost.

| | intent acc | macro-F1 | routing acc | esc. recall | esc. precision | missed esc. | over-esc. | reply safety-clean |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.164 | 0.035 | 0.700 | 0.000 | — | 66 (100%) | 0 (0.0%) | 1.000 |
| simple | 0.550 | 0.541 | 0.823 | 0.470 | 0.886 | 35 (53.0%) | 4 (2.6%) | 0.973 |
| **agent** | **0.805** | **0.797** | **0.873** | **0.636** | **0.913** | **24 (36.4%)** | 4 (2.6%) | **1.000** |

**Per-class F1 (agent).** The taxonomy holds up evenly; nothing collapses.

| `delivery_issue` | `refund_return` | `product_or_app_issue` | `prime_membership` | `order_status` | `account_access` | `payment_billing` | `feedback_or_other` |
|---|---|---|---|---|---|---|---|
| 0.857 | 0.840 | 0.840 | 0.800 | 0.769 | 0.769 | 0.750 | 0.750 |

Two shapes worth naming. `account_access` has perfect recall and the worst
precision (0.625) — the agent over-assigns it, which given the always-escalate
policy becomes wasted human time rather than risk. `payment_billing` is the
mirror: precision 0.938, recall 0.625, mostly lost to `refund_return`. That is
the one taxonomy boundary I would redraw.

### Reply quality — blind human rating

75 replies, 25 per system, shuffled, **producing system hidden** while rating,
rated before the judge was run. The question was the one the system acts on:
*would I let this go out with no human reading it first?*

| | human says send |
|---|---|
| trivial | 12 / 25 (48%) |
| simple | 13 / 25 (52%) |
| **agent** | **24 / 25 (96%)** |

This is the most convincing result here because it does not depend on the judge.
It also exposes how weak the automated reply metric is: **the trivial baseline
scores a perfect 1.000 "safety-clean" while a human refuses to send half its
replies.** Regexes catch fabrication; they do not catch sending *"I'm sorry for
the trouble, please contact us"* to someone praising Alexa.

Worth noting the simple baseline — which copies the brand's *real* replies —
scores 0.973, not 1.000: four of AmazonHelp's own replies promise a specific
outcome and two contain live URLs. **Copying what the brand actually did is not
automatically safe.**

### The LLM judge, and why I do not trust it

A four-dimension rubric (relevance, grounded, tone, actionable) plus one binary
verdict, which is the only one used for agreement.

| | relevance | grounded | tone | actionable | send without review |
|---|---|---|---|---|---|
| **agent** | 4.11 | **4.85** | 4.23 | 3.82 | 57.7% |
| simple | 3.05 | 4.53 | 3.59 | 2.85 | 26.8% |

**Agreement with a human is poor.** On the 75 blind-rated replies: raw agreement
0.627, chance agreement 0.522, **Cohen's kappa 0.218** — slight-to-fair, barely
better than chance. 17 disagreements are the judge being too strict, 11 too
lenient.

**And it is worse than a low number — the judge inverts the ranking:**

| system | human says send | judge says send |
|---|---|---|
| agent | **24/25 (96%)** | 14/25 (56%) |
| simple | 13/25 (52%) | 7/25 (28%) |
| trivial | 12/25 (48%) | **22/25 (88%)** |

The judge rates the **trivial baseline highest** — the system that answers every
message, including praise and jokes, with the same canned template. **Had I
trusted the judge, I would have shipped the worst system.**

The mechanism is visible in its own stated reasons — approving the template as
*"generic but safe, directs the customer to a private channel"*, rejecting a
correct agent reply for *"lacks clear next step or handoff"*. My rubric taught it
to check for an apology, a next step and a link: surface features a canned
template has by construction. It cannot tell that a handoff link is right for a
lost parcel and wrong for a compliment.

Consequently **the reply-quality claim in this report rests on the blind human
ratings, not the judge.** The one judge dimension I still credit is `grounded`
(4.85), because it tracks the regex-checkable property the automated harness
independently confirms.

---

## 5. What is misleading about my headline number

The headline is **macro-F1 0.797**. Here is everything wrong with quoting it.

**1 — It is measured on a distribution that does not exist.** The golden set is
stratified, so its class mix is not live traffic. Reweighting per-class recall to
the natural prior estimated over the held-out pool:

| | stratified (reported) | reweighted to live prior |
|---|---|---|
| trivial | 0.164 | **0.428** |
| simple | 0.550 | 0.596 |
| agent | 0.805 | 0.826 |

**On live traffic, answering `delivery_issue` to everything is right 43% of the
time**, because that genuinely is 43% of AmazonHelp's volume. My stratified
number makes the do-nothing baseline look 4× worse than production would. The
agent's lead over "do nothing" narrows from 0.77 to 0.40.

**2 — Intent accuracy is not the number that decides deployment.** 36.4% of the
messages that needed a human did not get one. Twenty-four customers — reporting
theft, chasing a two-month refund, on their fourth damaged delivery — would have
received an automated acknowledgement and nothing else. Routing accuracy of 0.873
sounds fine and hides the same thing: the trivial baseline scores 0.700 while
missing *literally every* escalation.

**3 — n = 220, so the error bars are wide.** A 95% Wilson interval on 0.805 at
n=220 is roughly ±0.05. Per-class it is far worse: `prime_membership` has **nine
examples**, so its 0.800 F1 is one example away from 0.72 or 0.87. The
trivial-vs-agent gap is unambiguous; a 0.02 difference between two tuned variants
would be noise.

**4 — Reproducibility was only verified at the end, and it failed the first
time.** Running the pipeline as the README instructs initially produced
*different numbers* — labels were keyed by position in the sampled file, and
expanding the escalation rules after labelling changed which messages were
sampled, silently mislabelling 88 of 220. Fixed (labels keyed by conversation id,
sample frozen, mismatches now assert) and both baselines reproduce exactly. But
the guarantee is hours old, not something the project had throughout.

**5 — I wrote the taxonomy, the labels, the prompt and the ratings.** One
labeller, no inter-annotator agreement. The classes look clean partly *because* I
defined them around what I saw while labelling. Blind rating removes
system-identity bias, not shared-criteria bias.

**6 — "Safety-clean 1.000" measures only what a regex can see.** It means no
invented URLs, order numbers, amounts or promised timeframes. It does not mean
the replies are good — the trivial baseline also scores 1.000.

**7 — 41% of real traffic is excluded before scoring begins.** Every number is on
English messages only. On live traffic the human queue is far larger than the
20.9% escalation share suggests — closer to half.

**8 — The LLM judge does not support any claim in this report** (kappa 0.218, and
it ranks the worst system highest).

---

## 6. Failure analysis — top 5

Computed by `src/failures.py`, not cherry-picked. Totals: 43 intent errors, 24
missed escalations, 4 over-escalations.

**1. The `repeat_contact` regex misses paraphrase — 14 cases, the largest single
failure.** This matters most because `repeat_contact` is the biggest escalation
class (31 of 66) and these are customers *already* failed once.

> *"After discussing with CSR over phone that return will be done…"* → predicted **auto**
> *"why is cust serv telling me they can't get me the book?"* → predicted **auto**

*Hypothesis:* my pattern enumerates surface forms (`contacted|emailed|called`)
and misses `CSR`, `cust serv`, `chat rep`, `telling me`. The architecture makes
it worse: rules run before the model and the model may only *add* escalations,
but nothing in its prompt tells it repetition is the brand's main escalation
trigger. **The fix is not a longer regex** — it is putting the escalation policy
in the prompt and keeping the regex as a floor.

**2. Catalogue requests read as product faults — 7 cases.**

> *"will you ever upload Law&Order SVU on prime videos?"* → `product_or_app_issue`, gold `feedback_or_other`

*Hypothesis:* my taxonomy's fault, not the model's. The model sees Prime Video
plus a complaint shape. The label name fights the definition — `feedback_or_other`
sounds like a residual bin. Splitting out `feature_request` would fix most.

**3. Fraud and theft with no fraud vocabulary — 4 missed escalations.**

> *"I've been charged £7.99 but my account is closed"* → **auto**
> *"Received order for OnePlus 5T **without the handset in it**"* → **auto**,
> and the drafted reply only said *"please don't provide order details"*

*Hypothesis:* both describe theft without using `fraud|stolen|unauthorised`.
The second is also the worst reply in the set — the model copied a boilerplate
line from a retrieved precedent where it was apt. **Precedent grounding transfers
boilerplate as readily as substance.**

**4. `order_status` vs `delivery_issue` — 6 cases, both directions.** My own rule
is "has the promised date passed?", which is often undecidable from the message
alone. **This is measurement noise, not model error — part of the reported error
bar is my own labelling inconsistency.**

**5. Seller and brand disputes read as account problems — 4 cases.** "Blocked
out", "unauthorized", "my listing" read as account compromise. Both routes
escalate, so routing is right for the wrong reason — but the reason is the
payload a human triager acts on. Worth noting **10 of the agent's escalations
came from the always-escalate `account_access` policy, not from model judgement.**

*Also observed:* retrieval leaks the `2/2` halves of multi-part brand tweets into
precedent, which occasionally surface as incoherent fragments. A one-line corpus
filter I did not get to.

---

## 7. What I would do next with one more week

In priority order, by expected effect on the number that matters.

1. **Move escalation reasoning into the model, keep the regex as a floor.** The
   largest failure (14 missed `repeat_contact`) is regex brittleness against
   paraphrase. Put the policy in the prompt; let the model apply the *concept* of
   "already sought help and was not helped".
2. **Second labeller on the 220, report kappa on intent.** Until that exists,
   every number has an unmeasured component that is just me. Poor agreement on
   the `order_status`/`delivery_issue` boundary would be a taxonomy fix, not a
   model fix.
3. **Split `feedback_or_other`** into `feature_or_catalogue_request` and
   `no_action_needed`, and rename it so it stops attracting residual predictions.
   Addresses failure modes 2 and 5 — 11 of 43 intent errors.
4. **Rewrite the judge rubric** to force an explicit "is this appropriate *to this
   specific message*?" gate before any box-ticking, and re-measure kappa. A judge
   at 0.218 is not an instrument.
5. **Calibrate confidence.** The agent emits 0.9 on almost everything including
   its mistakes, so the 0.55 floor currently escalates nothing. Either extract a
   real signal (logprobs, self-consistency) or drop the mechanism and stop
   claiming it works.
6. **Clean multi-part tweets from the retrieval corpus.**
7. **Embedding retrieval, measured against TF-IDF rather than assumed better.**
   Precedent relevance is never directly scored today; it should be its own
   metric before swapping retrievers.
8. **Then non-English**, starting with Spanish and Portuguese, which have enough
   corpus volume to retrieve precedent natively.

---

## 8. A note on what this project actually taught me

Three bugs shaped this work more than any modelling decision, and they share a
shape worth stating plainly.

| bug | effect | how it was caught |
|---|---|---|
| Handle-filtering broke thread reconstruction | 57% of "opening messages" were mid-thread fragments | reading a sample of real messages |
| Reasoning-model responses returned empty and were cached | ~6% of calls scored as agent failures | smoke-testing the judge on 3 examples |
| Labels keyed by position, not id | 88 of 220 silently mislabelled | running the pipeline from scratch |

**Every one produced plausible numbers while being wrong, and not one threw an
error.** Each was caught only by checking output against something independent —
a sample of real data, a three-example test, a from-scratch re-run. None would
have been caught by more modelling.

That is the argument for the evaluation harness being the deliverable rather than
the agent, and it is why this report leads with missed escalations, reweights to
the live prior, and reports that my own judge failed rather than quietly dropping
it.

---

*Reproduction: [README.md](README.md) · Decisions: [DECISIONS.md](DECISIONS.md) ·
Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(Kaggle, `thoughtvector`) · Model: Llama 3.3 70B Instruct via OpenRouter ·
Banking77 was not used — the intents here are defined from AmazonHelp's own
traffic. AI coding assistance was used throughout, per the assignment's rules.*
