# Decision log

The non-obvious calls, and why. Roughly in the order they were made.

---

**1. Brand: AmazonHelp.** Largest in the dump by a wide margin — 169,840 outbound
tweets vs 106,860 for the runner-up (AppleSupport). Picked for volume, because
the retrieval-grounding idea only works if there is enough history to retrieve
*from*, and because Amazon's queue spans delivery, payments, subscriptions,
devices and digital content, so the intent taxonomy has to be real rather than
two labels wearing a hat.

**2. Threads reconstructed from the reply graph, not from the `@AmazonHelp`
handle.** My first version did one pass and kept rows where the brand wrote the
tweet or the text contained `@amazonhelp`. It looked fine and was quietly
broken: only 43% of the extracted "first customer messages" were actually the
head of their thread. Customers often open with `Amazon is useless` and no
handle at all, and the brand picks it up — so handle-filtering decapitates the
conversation and you end up classifying mid-thread fragments like *"That would be
AMZL US. didn't even stop."* Rebuilt as two passes: build a union-find over
`in_response_to_tweet_id` for all 2.8M tweets, then keep only components that
AmazonHelp took part in. True-opener rate went 43% → 98.7%. This was the single
highest-leverage fix in the project and it was invisible in every summary
statistic.

**3. The unit of work is (thread-opening customer message → brand's first
reply).** Not the whole multi-turn thread. The three questions the assignment
asks — classify, draft, route — are all decisions made *on arrival*, before any
back-and-forth exists. Training or evaluating on later turns would leak
information the agent will not have at inference time.

**4. Eight intents, derived from the data rather than declared up front.** Term
frequencies over 80k openers produced seven recurring actionable clusters; ~21%
of messages matched none of them, and rather than force them somewhere, that
became `feedback_or_other` — a real class covering praise, rants, jokes and
catalogue requests. It is 20% of the golden set, so pretending it does not exist
would have been a 20% error floor.

**5. A written PRIORITY order for multi-intent messages.** 35% of messages match
more than one intent's keywords ("my Prime delivery is late and I want a
refund"). Single-label classification without a stated tie-break is unmeasurable
— two reasonable people label it differently and the disagreement looks like
model error. Order is `account_access > delivery_issue > refund_return >
payment_billing > prime_membership > product_or_app_issue > order_status >
feedback_or_other`, ranked by what the brand must act on first: a compromised
account blocks everything else, money already taken outranks money owed back.

**6. Four tie-break rules written down *during* labelling, not before.** Package
late/lost/damaged-outside → `delivery_issue`, but contents wrong/missing/faulty
inside a box that did arrive → `product_or_app_issue`. Catalogue *requests* →
`feedback_or_other`, but content promised and missing → `product_or_app_issue`.
A charge for the Prime fee itself → `prime_membership`, any other unexpected
charge → `payment_billing`. I only discovered these were ambiguous by hitting
them repeatedly at rows 34, 111, 139 and 203; they are now in `src/intents.py`
so the model is held to the same rule the labels were.

**7. Golden set stratified, not random.** A random 220 would contain ~12
`account_access` and one or two of most escalation triggers — too few to measure
anything per-class. So quotas per intent from a cheap keyword prior, plus a
quota of messages the prior *cannot* place, plus a quota that trips an
escalation rule. The prior only chooses what to read; every label was assigned by
hand against the written definitions, and the prior's guess was not shown while
labelling. The cost is that the class mix is not live traffic — hence macro-F1
as the headline and a reweighting in the README's "misleading" section.

**8. 20% of conversations held out before anything else touches them.** The
golden set is sampled only from that held-out pool, so no evaluated message can
retrieve its own historical reply as precedent. Split is by MD5 of the
conversation id, not `hash()`, which is salted per process and would silently
reshuffle the split between runs.

**9. "Auto-handled" means *safe to post unreviewed*, not *resolved*.** This is
the framing decision the whole routing metric rests on. AmazonHelp's real public
replies almost never resolve anything — they acknowledge and hand off to DM,
phone or a help page. Defining auto-handling as "the agent fixes it" would make
the honest score approximately zero and would measure the wrong thing.

**10. Escalation is rule-first, and the LLM may only *add* escalations, never
remove them.** Eight hard regex rules (legal, fraud, safety, vulnerability,
repeat contact, PII, seller dispute, abuse) fire before the model is consulted
and cannot be overridden. The errors are not symmetric: auto-replying to a fraud
report or a legal threat is a different kind of event from making an agent spend
thirty seconds on a routine one. A model that can talk itself out of escalating
is strictly worse than a regex that cannot.

**11. `account_access` always escalates, by policy, regardless of wording.** The
standard reply for account problems invites credentials and account details into
a public Twitter thread. This deliberately over-escalates benign questions like
*"can you hide orders from others on the account?"* — a real cost, visible in the
failure analysis, accepted on purpose.

**12. The repeat-contact threshold is ≥3 occurrences, or explicit prior contact
that is still unresolved.** Two failures with no prior contact stays auto. This
was tuned while labelling, not after seeing scores: "second time this week" is
the single most common delivery complaint in the corpus, and escalating it would
put roughly half the queue in front of a human, which is the same as having no
triage at all.

**13. TF-IDF retrieval (word 1-2 grams + char 3-5 grams), not embeddings.** The
messages are short, misspelt and dense with product nouns; char n-grams absorb
the typos, and the whole index builds in about a minute on a laptop with no
model download, which is what "reproduce in under 15 minutes" actually requires.
Order and tracking numbers are collapsed to a `<num>` token because they are
unique per customer and only dilute the vocabulary. An embedding index is the
obvious next upgrade, not a prerequisite.

**14. One LLM call does classification and drafting together.** The retrieved
precedents are the expensive part of the prompt and both steps need them.
Splitting into two calls doubled cost for no accuracy gain worth reporting.

**15. The "simple" baseline copies the nearest historical reply verbatim.** This
is a deliberately strong baseline, not a strawman. The brand's replies are
heavily templated, so copy-the-nearest-neighbour is genuinely competitive — and
if a generated reply cannot beat it, generation is not earning its cost. Worth
noting what this exposed: the brand's own real replies fail my automated safety
checks (promised timeframes, live URLs), so "do what the brand did" is not
automatically safe.

**16. Both baselines run with no API key.** Only the agent and the judge need
one. A reviewer with no key can still reproduce the data pipeline, the golden
set and both baselines, which is most of the evidence.

**17. Judge-human agreement is measured on the binary "send without review", not
on the 1-5 rubric scores.** The binary is the decision the system actually acts
on, and agreement on a 5-point scale would flatter the judge. Cohen's kappa is
reported alongside raw agreement because the classes are unbalanced — a judge
that says "send" every time scores well on raw agreement and ~0 on kappa.

**18. The judge rates a blind mix of replies from every system.** Shuffled, with
the producing system hidden from the rater, so the judge is validated across the
full quality range it will see rather than only on the good replies.

**19. LLM calls are disk-cached and run at temperature 0.** Re-running the
evaluation is free and returns byte-identical numbers. An evaluation harness
whose results move when you run it twice cannot be used to compare two systems.

**20. Non-English is detected and escalated, not translated.** 41% of AmazonHelp
openers are not English (Spanish, Portuguese, Japanese, Italian, Arabic). Doing
that properly means per-language precedent retrieval and per-language quality
evaluation, which I could not evaluate honestly in the time available. The gate
is a crude ASCII-ratio + function-word heuristic and it leaks — two German and
Spanish messages reached the golden set and are labelled as the failures they
are. Scoping this out is the largest single thing I chose not to build.

**21. Empty LLM responses are a retryable failure, never a cached result.**
Found by smoke-testing the judge on three examples before running it on 220.
Groq's `gpt-oss-120b` is a reasoning model: its chain of thought goes into a
separate `reasoning` field that still counts against `max_tokens`, so on a long
prompt it spent the whole budget thinking and returned an **empty** `content`
with `finish_reason: "stop"` — which looks like a successful call. My client had
cached those empties and the agent scored them as `agent_error`. 9 of 157 cached
responses (~6%) were affected, and they would have been reported as agent
failures rather than as my token budget being too small. Fixes: send
`reasoning_effort: "low"` (measured 40% fewer completion tokens), treat empty
content as retryable with a doubled budget, and include `reasoning_effort` in the
cache key so results stay reproducible. The general lesson is that a silent
failure that resembles success is worse than a crash, and the only reason I
caught it was testing a component on three examples before trusting it on 220.

**22. Provider chosen by measurement, not by default.** Groq's free tier is
capped at 8,000 tokens/minute, and with ~1.8k-token prompts that is roughly four
requests per minute — the agent run stalled in 90-second blocks. Benchmarking
OpenRouter on the same prompt size gave 6 calls in 42 seconds with no limiting,
about 3x faster end to end, so the reported results use OpenRouter with Llama
3.3 70B Instruct. Llama 3.3 is also a non-reasoning model, which removes the
empty-content failure in #21 at the source instead of working around it. Both
keys stay in `.env`; switching is one line.

**23. Duplicate keys in `.env` warn instead of silently resolving.** The file
ended up with `LLM_PROVIDER` twice and the loader took first-wins, so a
configured OpenRouter key sat unused while everything ran against Groq. Silent
precedence in config loading is a bad trade: the loader now prints a warning and
keeps the first value, so the behaviour is unchanged but visible.

**24. Reporting that my own LLM judge failed, rather than quietly dropping it.**
Cohen's kappa against a human is 0.218 and the judge ranks the *trivial* baseline
highest of the three systems — it rewards the presence of an apology, a next step
and a link, which a canned template has by construction, and cannot tell that a
handoff link is right for a lost parcel and wrong for a compliment. The tempting
move was to tune the rubric until the numbers looked respectable and report only
that. Instead the failure is the headline of the judge section, the reply-quality
claim was moved onto 75 blind human ratings, and the judge's send-rate is marked
as not usable for comparing systems. A judge validated against nothing is
decoration; this one was validated and it failed, which is a result.

**25. The human ratings were produced blind and before the judge ran.** 75
replies, 25 per system, shuffled with the producing system hidden. Rating after
seeing the judge's verdicts, or knowing which system wrote which reply, would
have made the agreement number meaningless. The rating rule was written down
first (`src/_judge_human.py`) and includes the deliberate asymmetry that a bland
generic reply is acceptable on a real complaint and unacceptable on a joke --
the failure being measured is inappropriateness, not blandness.

**26. The response cache turned a network outage into a non-event.** DNS
resolution failed partway through the judge run and the process died after
exhausting retries. Because every successful call was already on disk, re-running
resumed from where it stopped at no cost. This was not why the cache was built --
it was built for reproducibility -- but it is the second time the design paid for
itself, and it is the reason a rate-limited hour-long evaluation is safe to run
on a laptop.

**27. The golden set is a frozen artifact, and labels are keyed by id, not by
position.** Caught by running the pipeline the way the README tells a reviewer to
run it, which produced different numbers than I had reported (simple baseline
macro-F1 0.431 instead of 0.541). Cause: `_golden_labels.py` keyed the 220 labels
by index into `golden_unlabelled.jsonl`, while `build_golden.py` selects its
escalation stratum using `ESCALATE_RULES` -- and I expanded those rules after
hand-labelling. Re-running therefore drew a different sample and slid 88 of 220
labels onto the wrong messages, silently. Fixes: labels keyed by
`conversation_id`; `finalise_golden.py` asserts on any unmatched id so a
mismatch crashes instead of mislabelling; `data/golden_unlabelled.jsonl` is
committed as the frozen set of messages that were actually labelled, and
`build_golden.py` documents provenance rather than being part of reproduction.

This is the third bug of the same family in this project -- thread
reconstruction keeping 43% true openers, empty LLM responses cached as model
failures, and index-keyed labels -- and the pattern is worth stating plainly:
**every one produced plausible numbers while being wrong, and none of them threw
an error.** The only reason any was caught is that the output was checked against
something independent: a sample of real messages, a three-example smoke test, and
a from-scratch re-run. That is the argument for the evaluation harness being the
deliverable rather than the agent.
