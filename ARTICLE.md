# When "Probably Correct" Isn't Good Enough: Statistical Guarantees for LLM Document Extraction

## The problem everyone has

Walk into any company in medical, insurance, wealth management, or finance and you will find the same thing: mountains of unstructured documents. Claim forms scanned as TIFFs, investment memos as PDFs, pitch decks as PPTX, lab reports as PNGs. Buried inside them is structured information the business runs on — policy numbers, diagnoses, portfolio allocations, contract dates, revenue figures.

Today that information is extracted by people. It is slow, expensive, error-prone, and it does not scale. Large language models — especially agentic pipelines built around them — are an obvious fit: they read anything, they understand context, and they can emit structured output directly against a schema.

So why isn't every back office already automated? Because of one question that stops every procurement conversation:

**"How do you guarantee the extracted output is correct?"**

A single LLM call gives you an answer, not a guarantee. The model may hallucinate a value, misread a table, pick the wrong date out of three candidates, or confidently invent a field that isn't in the document at all. In domains where a wrong policy number or a wrong dosage has real consequences, "the model is usually right" is not an acceptable answer.

## The core idea: treat extraction as a repeated experiment

You cannot make a single stochastic process reliable. But you can make *many* stochastic processes measurable. The approach rests on three mechanisms.

### 1. Multi-call

Ask the same model the same question several times. LLM sampling is non-deterministic; that is usually seen as a weakness, but here it becomes a measurement instrument. If a model extracts `policy_number = "PX-48812"` in five out of five independent calls, that stability itself is evidence. If the value flips between calls, you have just detected uncertainty that a single call would have silently hidden from you.

### 2. Multi-model

Repeat the experiment across *different* models. Different model families have different training data, different failure modes, and different biases. When two unrelated models independently converge on the same value, the probability that both hallucinated the same wrong answer drops sharply. Model diversity is the extraction-world equivalent of getting a second opinion from a doctor who trained at a different school.

Both mechanisms collapse into one primitive: a fan-out of `N models × M calls` independent structured-output requests, executed concurrently, with partial failures tolerated. The output is not one answer — it is a *distribution of answers*.

### 3. Agentic aggregation

Raw votes are not enough, because language models phrase the same fact in different ways. One call returns "McKinsey Global Institute", another returns "MGI". One says "$29 trillion to $48 trillion", another says "29–48 trillion USD". A naive string-equality vote would fragment these into separate candidates and destroy your consensus signal.

This is where the agentic layer comes in: an LLM is used *on the aggregation side* to cluster semantically equivalent values before counting. The pipeline first groups exact matches cheaply (normalize whitespace and case), and only when genuinely distinct variants remain does it ask a model: "which of these strings describe the same fact?" The result is a set of semantic groups, each with a canonical value and an occurrence count.

## What you get: reliability as a number

After fan-out and semantic aggregation, every extracted field carries hard statistics:

- **count** — how many independent calls produced a semantically equivalent value
- **total_calls** — how many calls succeeded overall
- **score** — count / total_calls, the field's consensus ratio
- **candidates** — every competing value group, with its own count

And that gives you an operational decision rule instead of blind trust:

| Consensus | Interpretation | Action |
|---|---|---|
| Value appears in **all** calls (score ≈ 1.0) | Models independently agree | Accept automatically |
| Value appears in **most** calls | Likely correct, some ambiguity | Accept with flag, or lightweight verification |
| Value appears in **few** calls | Genuine uncertainty or hallucination | Route to investigation — an additional adjudication step (LLM verdict call or human review) |
| No value in any call | Field absent from document | Return null with confidence |

The critical property: **the system knows when it doesn't know.** Instead of one opaque answer, you get a calibrated signal that separates the 90% of fields that can flow straight through from the 10% that deserve human eyes or a deeper agentic investigation loop. That split is exactly what makes automation economically viable in regulated domains — you don't need zero human review, you need human review *concentrated where it matters*.

## The extra trick: grounding

Statistics tell you models agree. Grounding tells you *why*.

Every extraction call is forced to return, alongside each value, a short verbatim quote from the source document that supports it. In the schema this is a first-class field:

```json
{"name": "revenue_projection",
 "value": "$29 trillion to $48 trillion",
 "grounding": "generate $29 trillion to $48 trillion in revenues by 2040"}
```

Grounding does three jobs at once:

1. **It suppresses hallucination at generation time.** A model required to quote its evidence is far less likely to invent a value — there is nothing to quote.
2. **It makes verification mechanical.** A grounding quote either appears in the document or it doesn't. A value whose grounding cannot be found in the source text is automatically suspect, regardless of how many calls agreed on it.
3. **It gives auditors a trail.** In insurance and medical contexts, "where did this number come from?" must have an answer. Grounding is that answer, attached to every field, preserved through aggregation so every candidate group carries its supporting quotes.

Consensus without grounding can still mean "all models made the same plausible mistake." Consensus *with* verifiable grounding is about as close to a guarantee as generative extraction gets.

## The architecture, minimally

The reference implementation in this repository keeps the whole idea to a handful of small components:

```
document (PDF/PPTX/TIFF/...)
   → loader          normalize into text or page images
   → fan-out         N providers × M calls, concurrent structured-output requests
   → per-call result typed schema: {field, value, grounding}
   → semantic merge  exact pre-grouping + LLM clustering of equivalent variants
   → result          per field: canonical value, count, score, all candidates, groundings
```

No heavyweight agent framework is required. Structured output is enforced by passing a JSON schema to the model and validating the response against typed models (Pydantic), with a retry on validation failure. The fan-out is a page of asyncio. The semantic merge is one focused LLM call per contested field. The entire "guarantee machinery" is maybe five hundred lines of code — the value is in the *shape* of the pipeline, not its size.

## Takeaways

- Single-call LLM extraction produces answers; repeated, diversified extraction produces answers **with error bars**.
- Multi-call exposes model instability; multi-model defends against correlated failure modes; together they turn correctness into a measurable frequency.
- Semantic aggregation is essential — without it, formatting noise destroys the consensus signal. Use a model to merge variants, then count.
- Consensus score is a routing signal: full agreement flows through, partial agreement gets flagged, low agreement gets investigated. Humans review the residual, not the firehose.
- Grounding — forcing the model to quote its evidence — is cheap, mechanically verifiable, and belongs in every extraction schema from day one.

The manual back office isn't replaced by a smarter model. It is replaced by a *humbler system* — one that measures its own reliability and says, field by field, "trust this, check that."
