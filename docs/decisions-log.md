# Decisions

Choices and absences that are not obvious from the code alone. Numbers stay
stable; removing an entry leaves a gap.

**D7 — schema changes rebuild the index.** `Store.open()` recreates the
database when `schema_rev()` changes; there are no migrations.

**D10 — content categories are part of index identity.** Discovery resolves
`file.content_type` from the configured category rules, and `categories_rev()`
forces a rebuild when those rules change.

**D18/D20 — `.repoglassignore` composes with `.gitignore`.** `hard_exclude`
prunes first; `.gitignore` and `.repoglassignore` are then combined with
gitignore syntax, with `.repoglassignore` taking the last match.

**D21 — the default index lives outside the repository.** The indexed tree stays
read-only; `data_dir` is the opt-in override.

**D22 — default lexical text is derived, not stored per chunk.** `chunk_fts`
reads the `chunk_lexical` view. `chunk.lexical_override` is written only when
the configured lexical rendering differs from the default path-plus-body form.

**D25 — there is no derived confidence field.** `Hit.tiers` exposes which
retrievers matched; callers should not treat a rescaled score as a probability.

**D12 — raw source and query-shape weighting are one change, not two.**
Raw text alone is *worse* than the distillation it replaced. The gain is
entirely in the weighting, so adopting either half on its own measures a
regression and reverts the wrong thing.

---

## Tried and rejected

A record, not a prohibition. Each was built and measured; the reason is
kept because rebuilding a thing to re-learn why it went is the waste
this list exists to prevent.

| Tried | Why it went |
|---|---|
| `tier_routing` — prose queries to the vector tier alone | Lost more than it gained, and took any-hit down with it. |
| Window-only chunking | Loses to `hybrid` everywhere; byte windows cut across markdown headings. |
| Prose distillation (`distill_docs`) | Docs get worse. Code appears to improve only because degraded doc chunks stop competing. Kept, default off. |
| Path penalties as a general rule | Harmful under `content="all"`, a no-op under `content="code"` — the content filter had already removed what they targeted. |
| A confidence score derived from rank or magnitude | See D25. |
| Capturing imports as references | A file that imports a name almost always calls or annotates it too, and both are already captured; what imports would add is re-export shims. Annotations were measured alongside and adopted — for a type never instantiated they are its only uses. |
| A call graph, and PageRank over it | The graph is derivable, but resolving a callee by name is mostly noise: one constructor call cannot mean every constructor, and in-degree promotes leaf utilities — the opposite of what an architecture question wants. Scope-aware resolution is the fix and is buildable, but same-file and unique-name resolution together reach 23% of references over 8 repositories and 6% on JavaScript, so what is left is still too sparse to propagate without per-language import parsing. |

**A caution that applies to the whole table.** Several of these were
first measured with `rerank` disabled, which was itself a defect: the
layer raised `NameError` on every call through the public API while unit
tests called into it directly and passed. Verdicts predating that fix
were taken on an instrument that was not running, and the piecemeal
rejections of semble's individual boosts were wrong for exactly that
reason.

A second caution, learned later: NDCG@10 scores only the positions of
labelled files, and most result lists are mostly unlabelled. A change
can restructure a large share of results and move the metric by nothing.
"Measured no difference" here means no evidence either way, not no
effect.
