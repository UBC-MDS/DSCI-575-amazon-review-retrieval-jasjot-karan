# Final Discussion

## 1.2 Experiment with the LLM

### Models Used
- **Model 1:** `microsoft/Phi-4-mini-instruct` (3.8B parameters) — via Ollama as `phi4-mini`
- **Model 2:** `meta-llama/Llama-3.2-3B-Instruct` (3B parameters) — via Ollama as `llama3.2:3b`
- **Model 3:** `Qwen/Qwen2.5-3B` (3B parameters) — via Ollama as `qwen2.5:3b`

---

### Prompt

All five queries were run using identical retrieved context (top-5 documents via `HybridRetriever`) and the same prompt template defined in `src/prompts.py` as `SYSTEM_PROMPT_V3`.

---

### Outputs

| Query | Phi-4-mini-instruct | Llama-3.2-3B-Instruct | Qwen2.5-3B |
|---|---|---|---|
| charger cable that stops working quickly | Based on user reviews, I recommend [B07ZYLF9WY...] | Based on your query for a charger cable that s... | For a charger cable that stops working quickly... |
| chair that hurts your back after sitting | I recommend [B08ZSLRF2H] for its green backgro... | [Product rank: 1] [B08ZSLRF2H]\nThis product i... | Based on the provided product context, [B08ZSL... |
| speaker not loud enough for a room | Based on your requirement of needing an extrem... | For a speaker that is not loud enough for a ro... | For a speaker that is not loud enough for a ro... |
| laptop gets too hot when running programs | I recommend [B08CVQKQBX] for its excellent per... | [Product rank: 1] [B013BYA46G]\nThis product i... | For a laptop that frequently overheats during... |
| speaker loud bass | Based on your interest in speakers with powerf... | Based on the Product Context, I recommend [B07... | For speaker loudness and bass, I recommend [B0... |

---

### Key Observations

1. **Response Style:**
   - `Phi-4-mini-instruct` produces conversational, review-grounded responses that directly reference user needs before recommending a product.
   - `Llama-3.2-3B-Instruct` tends to lead with a ranked product ID and follows with a structured description, making outputs feel more list-like.
   - `Qwen2.5-3B` produces fluent, natural-language responses similar in tone to Phi-4, but with slightly less specificity in referencing review details.

2. **Relevance:**
   - All three models correctly identified relevant ASINs from the retrieved context across most queries.
   - For ambiguous queries (e.g. *"chair that hurts your back after sitting"*), Phi-4 and Qwen interpreted the query as a product-recommendation task (recommending ergonomic chairs), while Llama leaned more toward a literal product-rank listing.

3. **Consistency:**
   - `Phi-4-mini-instruct` showed the most consistent formatting across all 5 queries.
   - `Llama-3.2-3B-Instruct` occasionally included raw formatting artifacts (e.g. `\nThis product i...`) suggesting its instruction-following for structured output is slightly weaker at 3B scale.

---