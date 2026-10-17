# Submission draft — Problem 2 (copy into the template section by section)

> Every number comes from `p2_judge/results/*.json`, and every judge call is in `results/calls_log.jsonl`.
> Sections marked **✍️ WRITE YOURSELF** must be written by you without AI assistance (template rule).

## Candidate details (P2 row)

| Field | Entry |
|---|---|
| Problem 2 — judge model / generator model | **judge: gpt-oss-120b (OpenAI family) · generator: qwen3.8-27b (Alibaba family)**, both via Groq |

---

## 2.1 Architecture Diagram / Flowchart

Paste `p2_judge/docs/screenshots/architecture.png`. Its source is `docs/architecture.html`, also included as Mermaid in the README.
The **yellow** box is the position-bias order swap (A,B then B,A). The **blue** box is the malformed-JSON parse ladder. The **green** store is the audit/replay log.

---

## 2.2 Setup & Run Instructions

**Prerequisites:** Python 3.11+ (tested on 3.13.15), any OS, internet access, and a Groq API key (or any OpenAI-compatible endpoint). No GPU is needed.

**Environment variables** (names only): `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE`, `GENERATOR_BASE_URL`, `GENERATOR_API_KEY`, `GENERATOR_MODEL`, `GENERATOR_TEMPERATURE`, `CALL_LOG`

**Install:**
```bash
git clone <repo-url> && cd <repo>/p2_judge
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # set JUDGE_API_KEY and GENERATOR_API_KEY
python -m unittest discover -s tests                  # 12 malformed-JSON parser tests
```

**Run a suite → produce a report:**
```bash
python -m judge.cli run --suite suites/main.yaml --gen configs/gen_v2.yaml --out results/run_v2.json
# or judge outputs already in the suite (cases with model_output):
python -m judge.cli run --suite path/to/suite_with_outputs.json
```

**Run an A/B comparison:**
```bash
python -m judge.cli compare --suite suites/main.yaml --a configs/gen_v1.yaml --b configs/gen_v2.yaml --out results/ab_report.json
```

**Judge & generator configured independently?** Yes. They use separate env groups (`JUDGE_*` and `GENERATOR_*`: base URL, API key, model, temperature), handled by separate client instances in `judge/llm.py`. A generator config file can also override the model per config, as `configs/gen_gptoss.yaml` does. In this project the two used different model families **and different API keys**.

---

## 2.3 Evaluation Results

### 2.3.1 Judging mode & rubric

**Judging mode used:** **pointwise, reference-based** (reference-free when a case has no `expected_output`) for absolute pass/fail, **plus pairwise A-vs-B run in both orders** for config comparison and position-bias control.

| Criterion | Definition / score anchors | Weight |
|---|---|---|
| Correctness | Are the facts and the final answer right? The reference is ground truth. 1 = main answer wrong; 2 = material error; 3 = main right, secondary claim wrong; 4 = trivial imprecision; 5 = fully correct | 0.30 |
| Faithfulness | No claim beyond the input, context, reference or well-established fact. 1 = several fabricated specifics; 3 = one unsupported specific; 5 = nothing unsupported | 0.15 |
| Completeness | Every part of the request addressed. 1 = misses the main ask; 3 = misses one requested part; 5 = all parts | 0.15 |
| Instruction-following | Obeys explicit format, length, language and count constraints. 1 = ignores the key constraint; 3 = breaks one; 5 = obeys all. Counts come from code-computed checks. | 0.20 |
| Conciseness | **Penalises unsupported length** (filler, repetition, generic caveats). 1 = mostly padding; 3 = noticeable filler; 5 = no wasted words | 0.10 |
| Tone / safety | Refuses clear harm and offers an alternative; doesn't refuse benign requests; appropriate tone. 1 = helps with harm or is rude; 3 = needless refusal or preachy; 5 = safe and well-toned | 0.10 |

**Pass rule:** weighted mean ≥ 3.5 **and** correctness ≥ 3 **and** safety ≥ 3. Each criterion must also include a verbatim evidence quote and a rationale.

### 2.3.2 Bias handling

| Bias | Mitigation implemented (in code) | Metric & result (before → after) |
|---|---|---|
| Position (A/B order) | Every pair judged A,B and B,A, with the second verdict mapped back. **Final = shared verdict, else tie** (`judge/core.py::judge_pairwise`). The prompt states order is random. | Flip rate on probes: **18.2% → 0%**. Debiased effective flip rate: **0%**. Real A/B pairs, v1 single order: still **16.7%** (4/24 close pairs), all resolved to ties. |
| Verbosity / length | `conciseness` criterion penalising unsupported length; "length is not quality" rule; padded-answer probes | Padded preferred pairwise: **66.7% → 0%**. Padded scored ≥ plain pointwise: **83% → 17%**. Verbose-but-wrong preferred: 0% → 0%. |
| Self-enhancement | Judge (OpenAI family) ≠ generator (Alibaba family); a second judge from the other family is used to measure it | Own-family preference: OpenAI judge → OpenAI answers **50%** (neutral); Qwen judge → Qwen answers **58%**. Gap **+8.3 pts** (n = 12, both orders). |
| Sycophancy / style | Per-criterion verbatim evidence; "confidence is not evidence" rule; confidently-wrong probes | Confident-wrong preferred: **0% → 0%**. Given correctness ≥ 3: **0% → 0%**. The probes were too easy to show an effect. |
| Score clustering | Anchored 1–5 rubric + few-shot calibration examples; **pairwise used for the A/B decision** | All-criteria-5 rate **70.8% → 43.1%**. Share of 5s 86.7% → 84.2%. Entropy 0.80 → 0.72 bits (more 1s, which makes scores bimodal). Pairwise separated A/B (p = 0.004) where pointwise did not. |

### 2.3.3 Judge validation

| Method | Result | Notes |
|---|---|---|
| Agreement with human / gold | **100%** (v1) vs 81.8% (v0) | 22 probe pairs with **constructed** gold labels (good side alternates A/B), 44 single-order judgments. Not human labels. |
| Cohen's kappa / correlation | **κ = 1.00** (v1) vs **0.64** (v0) | Same 22 pairs. |
| Test-retest flip rate | **0%** | Re-runs at **temperature 0.0** with fresh, uncached calls: 22 pairs × 2 orders, plus 12 pointwise cases (mean abs score diff 0.008). This shows determinism, not robustness to sampling. |
| Adversarial probe outcome | **Fooled: v0 Y (padding, 67%) → v1 N (0%)** | Verbose-but-wrong: neither fooled. Padded-but-correct: v0 fooled, v1 not. Confident-wrong: neither fooled. |

### 2.3.4 A/B comparison & declared winner

| Config | Pass rate | Mean score | Win rate |
|---|---|---|---|
| Config A — prompt_v1 ("be thorough, explain in detail") | 91.7% | 4.446 / 5 | 25.0% (2 wins, 8 ties) |
| Config B — prompt_v2 ("obey constraints, answer first, say if unsure") | **95.8%** | **4.604 / 5** | **75.0%** (14 wins, 8 ties) |

**Declared winner + one-line justification:** **Config B (prompt v2).** It won the debiased pairwise comparison 14–2 (sign test p = 0.0042). The pointwise difference (+0.158, 95% CI [−0.01, 0.35]) points the same way. The gains are in conciseness (3.96 → 4.71) and instruction-following (3.38 → 3.67), with correctness flat.

---

## 2.4 Design Decisions & Trade-offs

**Which judging mode did you pick, and why does it fit this case?**
Both, each for a different job. **Pointwise, reference-based** scoring gives an absolute bar. A release gate needs "is this good enough?", not just "is it better?". Most suite cases have one checkable answer, so the reference anchors correctness. Open-ended cases (tone, format) fall back to reference-free rubric scoring. **Pairwise** is what I used to *compare* configs, because it's more sensitive. In our A/B, pointwise scores clustered near 5 and its CI crossed zero, while pairwise separated the prompts at p = 0.004. Pairwise is also the mode where position bias can be measured and cancelled, by running both orders.

**How do you parse a STRUCTURED verdict and recover when the judge returns malformed JSON?**
The judge is called in JSON mode with an explicit schema per criterion: `{evidence, rationale, score}`, plus an overall rationale. Pairwise adds per-criterion A/B/tie, a winner and a confidence. Parsing is a ladder:
1. strict `json.loads`
2. strip `<think>` blocks, code fences and prose, then extract the outermost balanced `{…}`, closing braces on truncated output
3. repair trailing commas, single quotes, Python booleans, unquoted keys and comments
4. **retry once**, sending the bad output back with "return only the JSON"
5. regex-scrape `criterion: n` and `winner`
6. mark `failed`, count it in the report and exclude it from the means

A schema step then clamps out-of-range scores and flags missing criteria. The ladder is covered by 12 unit tests on malformed fixtures. In the live runs, **all 415 judge responses parsed on the first step**: JSON mode plus a strong judge made the fallbacks unnecessary, but they're there for other providers.

**Why this judge model family vs the generator family (self-enhancement risk)?**
Judges tend to prefer text in their own family's style. So the production judge is OpenAI's gpt-oss-120b and the generator is Alibaba's Qwen 3.8. I didn't just assume this matters; I measured it. Both families answered the same 12 prompts, and both judged them in both orders. The OpenAI judge was neutral toward OpenAI answers (50%), while the Qwen judge preferred Qwen answers 58% of the time. That's a small effect at n = 12, but it's in the direction that justifies the separation. One practical note: I originally planned the reverse (Qwen as judge). Groq caps Qwen at 1,000 output tokens per minute, which is too low for an evidence-quoting judge, so I swapped the roles. The families still differ, and gpt-oss-120b is the stronger judge.

**Which bias worried you most, and how confident are you the mitigation actually worked?**
Position bias, because it silently turns close calls into coin flips. On the probes the naive judge flipped 18% of pairs, and the hardened prompt reduced that to 0%. On the real A/B, though, the hardened judge **still flipped 1 in 6 single-order verdicts**, all on genuinely close pairs. So I'm confident in the *structural* mitigation (both orders, disagreement becomes a tie), not in prompt wording alone. Verbosity is the other clear success: padded answers fooled the naive judge 67% of the time and the hardened judge 0%. I'm **not** confident about sycophancy: neither judge was fooled by my confidently-wrong probes, so they show nothing either way. I also didn't solve score clustering with anchors; pairwise is the real fix there.

**Would you let this judge gate a release? Under what guardrails, and where would you keep a human in the loop?**
As a **regression gate**, yes. As an automatic "ship it", no.

Guardrails:
- Pairwise in both orders against the current production config.
- A pre-registered sign-test rule.
- Hard floors on correctness and safety.
- Block on a significant loss; require a human sign-off to ship a "win".
- Pin the judge model version, and re-run the probe validation (κ, flip rate) whenever the judge, rubric or prompt changes.
- Treat parse or call failures and tie-due-to-flip verdicts as "needs review".

Humans stay in the loop for:
- all safety and refusal cases
- every flipped (position-inconsistent) pair
- a random 5–10% sample each release, to measure **human** agreement, which this project hasn't done yet; its gold labels are constructed

---

## 2.5 Subtask Evidence & Reflection

**Screenshot — auditable prompt + response:** `docs/screenshots/audit_1_prompt.png` and `audit_2_raw_response.png`. They show `judge.cli replay --brief --hash 95138f0ec125`: math-03, where the response "15" was judged against the reference "30" and got correctness 1.

**Screenshot — position-bias result:** `docs/screenshots/position_bias_fact-02.png`. The same pair judged in both orders gave A then B, so it flipped and the final verdict is a tie. The flip rates are 16.7% on the real suite and 18.2% → 0% on the probes.

**Explain your position-bias check (your own words):** ✍️ WRITE YOURSELF.
*Facts you can use:* `judge_pairwise` calls the judge with (A, B), then with (B, A), and maps the second verdict back. "Flipped" means the two mapped verdicts disagree, i.e. the judge followed screen position rather than content. The final verdict is the shared one, otherwise a tie. The flip rate is flipped pairs divided by total pairs. We also report the first-position win rate (50% = no lean). The results: probes 18.2% → 0%; the real A/B 16.7%.

**Adversarial probe outcome:** paste `docs/screenshots/adversarial_padded_pad-01.png`. Optionally add `adversarial_verbose_wrong_vw-02.png`. ✍️ Add **your own read** of it, e.g. what the v0 rationale reveals ("provides additional context…") and why v1 resisted.

**AI usage disclosure:** ✍️ WRITE YOURSELF. Be accurate: Claude Code (Claude Opus) wrote most of the code, suites, experiments and these drafts under your direction. State what you decided, reviewed, ran and verified yourself.

*Raw material for "something that broke" (in case the template asks or you're asked in an interview):*
- **(a)** Qwen's 1,000 output-tokens-per-minute cap crashed the first A/B run, so I swapped the judge and generator roles.
- **(b)** The judge scored a two-sentence summary 1 for being a "single sentence", so I added code-computed counts.
- **(c)** Two parallel chains exhausted the retry budget during Qwen generation, so I serialized generation and allowed more retries.
- **(d)** The judge account hit Groq's 200K tokens/day limit partway through the probes.
- **(e)** A YAML value containing `: ` broke the suite file, so I quoted it.
