# Notes: design considerations for retrieval quality

Personal study notes — non-execution-log thoughts on what the metrics
actually mean, what numbers count as "good," and where we are vs.
published baselines. Written so I can re-read later and recover the
framing.

---

## Q1: What R@1 number is "good" for this task?

It depends on the downstream UX, not a generic threshold. Anchors:

| Use case | What R@K matters | Bar |
|---|---|---|
| User picks from a top-K grid (Pinterest, "show me 10 options") | R@10 | > 0.5 is enough; R@1 barely matters |
| Re-ranking stage (this pipeline is the candidate generator, downstream model picks the final top-1) | R@50 | > 0.7 is what counts |
| One-shot autonomous retrieval (chatbot returns one answer, no human in loop) | R@1 | > 0.6 minimum, ideally 0.8+ |
| Research benchmark (FashionIQ-style 6k gallery, paper headline) | R@10 | SOTA is ≈ 0.5–0.6 |

### Where we landed (Phase A, Marqo FashionCLIP)

On the 59,082-row FACap dress gallery with VLM-generated captions:
- R@1  = 0.258
- R@5  = 0.456
- R@10 = 0.533
- R@50 = 0.685

Read against the table:
- For a **top-10 grid UX**, R@10 = 0.533 is right at the threshold of
  "usable for a human-in-the-loop UI." Half the queries put the right
  dress in the user's first screen of results.
- For **re-ranking**, R@50 = 0.685 is below the 0.7 bar but close.
- For **one-shot autonomous retrieval**, we are nowhere near. R@1 needs
  to roughly double before this pipeline could replace human judgment.

### How does this compare to published numbers?

Most published CIR (composed image retrieval) systems land R@1 in the
0.10–0.25 range on FashionIQ's much smaller 6k gallery using direct
image+text fusion. We are getting comparable R@1 from a pure
caption-then-retrieve pipeline on a **10× harder gallery** (59k vs 6k).

That suggests two readings:
- The MiniLM anchor (R@1=0.084) was misleadingly weak — a poor encoder
  choice on the full 59k gallery looks much worse than a good encoder
  on a small benchmark gallery.
- The caption-then-retrieve recipe is more competitive than its
  reputation suggests *if* you pair it with a domain-pretrained encoder.

### Honest read for Phase A → Phase B

**R@10 = 0.53 is the headline number, not R@1.** It says: "if we show
the user a top-10 grid, the right item is in there for half the
queries." That is a usable starting point for a human-in-the-loop UI
*today*, and a strong launching pad for Plan_4 (contrastive
end-to-end).

The R@1 = 0.258 is the more honest "how often is this system fully
correct on its first guess" number — and it tells us the pipeline is
not yet at autonomous-retrieval quality. That gap is what Phase B is
designed to close.

---
