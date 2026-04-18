### Model Choice Justification 

We chose the phi4-mini-instruct model using Ollama as it provides strong reasoning and logic when outputting results based on user instructions compared to other models in its size range. The phi4-mini model is good at generating reasonable responses given user instructions and the system prompt, so we chose this model as we can clearly specify in our system prompt to the LLM to "not falsify any information" and only output product information that comes directly from the product metadata top-k results, and not from its own memory.

The phi4-mini model is also compute-constrained friendly, as its download size is only ~2.5 GB locally, which is minuscule compared to typical SSD memory sizes of 512 GB+ nowadays, and it typically uses only 4–8 GB of VRAM, which is supported by our laptops without being too computationally heavy.

The model is also very strong in reasoning ability as it is trained on fewer parameters (3.8B) compared to much larger models but has similar reasoning ability to them based on the HuggingFace docs. A possible downside is that since phi4-mini was trained on fewer parameters, it might be factually incorrect sometimes, but this is easy to fix as phi4-mini provides a tool-enabled function calling format where the user can instruct the model to call specific functions that use certain tools in the system prompt. This allows the output to the user be even more grounded, as we could include a tool such as a search engine like Tavily in the input context window to make sure the LLMs output matches what is said about the product on the web.

Last, the phi4-mini-instruct model also has a large context window and supports 128,000 tokens. This means that the combined total tokens of the input (user query, system prompt, previous chat history, and any attached documentation or embeddings such as product information) and the model's generated output of the context window from the model can be a maximum of 128K tokens, which is a large context window and provides enough space for us to pass a detailed system prompt along with long product metadata text for a large top-k to the phi4-mini model, meaning it can process a lot of product metadata information at once if top-k is high.


***Referenced HuggingFace docs for phi4-mini use cases: https://huggingface.co/microsoft/Phi-4-mini-instruct

--- 

### Choosing a Default System Prompt Variant

#### System Prompt 1 

```python
SYSTEM_PROMPT_V1 = """You are an Amazon shopping assistant answering customer questions about products using retrieved product metadata and reviews.

Mandatory rules. Any violation must result in immediate output rejection and reconstruction. No exceptions:

- Use ONLY the information provided in the Product Context below (under Product Context). Do not rely on prior knowledge, outside sources, or assumptions about products not listed explicitly.
- Every factual claim must be traceable to a specific product and that product's information in the context. Cite the corresponding ASIN in parentheses, e.g. (ASIN: B01ABC123), immediately after the claim.
- Do not begin the response with filler affirmations, compliments to the user, or restatements of the question. Move directly to the answer.
- Do not end the response with emojis, exclamations, or trailing questions.
- Do not produce generic template-like answers. Every sentence must reference specific products, features, prices, ratings, or review content from the context.
- Keep the answer concise, focused, and grounded, without any em dashes. Prefer a concise but informationally useful response that is formatted nicely for the end user; unless the question explicitly requires a longer comparison."""
```

##### Observations from System Prompt 1 

We found that SYSTEM_PROMPT_V1 (below) resulted in too much of a concise response from the large document metadata ranking corpus as V1 produced a single-paragraph answer that recommended **Rank 5** (the heca tech tenkeyless mechanical keyboard, ASIN B0765131GQ) while ignoring Ranks 1–4. The model defaulted to the product with the most context / metadata (the heca tech has a full specification sheet in `description`, while Ranks 1–4 have empty descriptions), but in doing so it overrode the semantic retriever's relevance ranking without any explanation. Therefore, we expected system prompt 1 to help the LLM output a grounded response where each product recommendation is ranked from best math to lowest math, however, it mixed up rankings and recommended the 5th ranked product instead of the 1st ranked product. Over different user queries, I also observed the same pattern of SYSTEM_PROMPT_V1 mixing rankings and recommending a worse ranked result from semantic retrieval compared to the best one.

#### System Prompt 2

```python
SYSTEM_PROMPT_V2 = """You are an Amazon shopping recommender answering customer questions about products using retrieved product metadata and reviews. Your goal is to return a ranked list of products from the Product Context below that match the user query. 

Mandatory rules. Any violation must result in immediate output rejection and reconstruction. No exceptions:

- Use ONLY the information provided in the Product Context below (under Product Context). Do not rely on prior knowledge, outside sources, or assumptions about products not listed explicitly.
- The Product Context is ordered by retrieval relevance, with [Product rank: 1] being the most relevant to the user's query and higher ranks being progressively less relevant. Prioritize higher-ranked products in your answer unless the metadata clearly shows a higher-ranked product does not match the user's needs. If you recommend a lower-ranked product, briefly explain why.
- Every factual claim must be traceable to a specific product and that product's information in the context. Cite the corresponding ASIN in parentheses, e.g. (ASIN: B01ABC123), immediately after the claim.
- Do not begin the response with filler affirmations, compliments to the user, or restatements of the question. Move directly to the answer.
- Do not end the response with emojis, exclamations, or trailing questions.
- Do not produce generic template-like answers. Every sentence must reference specific products, features, prices, ratings, or review content from the context.
- Keep the answer concise, focused, and grounded, without any em dashes. Prefer a concise but informationally useful response that is formatted nicely for the end user, unless the question explicitly requires a longer comparison."""
```

##### Observations from System Prompt 2

Once I changed SYSTEM_PROMPT_V2 to include a line about prioritizing higher-ranked products in its answe, the LLM output become more grounded and reflective of the true rankings of the semantic retrieval. The LLM produced a per-product breakdown for the top 3 retrieved products from semantic search (Ranks 1, 2, and 3), along with their ASIN citation, price, rating, and product descriptions that are correctly based on the retrieved context (for example, the motospeed price of $35.98 and the 4-star rating for an output both came directly from that products metadata). The 2nd system prompt then outputs a final recommendation for one product to the user based on the relevant highest ranked products, as it typically uses the 1st ranked product to recommend and backs up its recommendation with product metadaa information such as positive review feedback. For example, for the query: "mechanical keyboard good at coding": this system prompt justified its choice by referencing a review snippet that mentioned that the keyboard is fine if it is used for office work compared to gaming, which is a reasonable signal that the keyboard would be "good for coding." Therefore, since this system prompt does not hallucinate, cites ASINs consistenly, outputs multiple product options, and then chooses a best recommendation based on these top k options while also explaining its reasoning instead of automatically choosing the top product, it is more robust and grounded than the other methods. Since system prompt 2 has a higher chain of reasoning compared to the other outputs from other system prompts, it typically gives more grounded output compared to system prompt 1. 

However, V2 only outputted the top 3 out of the top 5 retrieved products instead of all 5 product rankings. For example, the product at rank 5 was dropped from consideration of being the best product to recommend, even though it had the most detailed product metadata (included a detailed description). Also, since V2 respects the semantic retrievers ranking and bases its product recommendation based on this, incorrect products can be recommended if the semantic retriever retrieves a weaker matching product as the highest ranked product. Another downside is that the output of V2 is pretty lengthy, which can lead to a lot of unneccessary token useage if our application were to scale up to a lot of users. 

#### System Prompt 3

```python
SYSTEM_PROMPT_V3 = """You are a strict Amazon shopping assistant. Answer the user's question in a concise manner using only the Product Context below in a maximum of 5 sentences.

Mandatory rules:

- Use ONLY the Product Context. Any claim not directly supported by the context must be omitted. 
- Cite the ASIN in square brackets for every product referenced, e.g. [B01ABC123].
- Every sentence must include at least one ASIN citation. Sentences without citations are not allowed.
- If the context is insufficient, respond with exactly this sentence and nothing else: "I do not have enough information to answer that. Please try searching another product."
- Do not begin with affirmations, compliments, or filler phrases.
- Do not end with emojis, exclamations, or questions.
- Do not produce template-like or generic answers. Every sentence must contain specific, context-grounded detail.
- Do not speculate about products outside the context, even if asked.
- Do not include any information outside the final answer. Do not explain your reasoning or reference the instructions.
"""
```

##### Observations from System Prompt 3

Once I changed SYSTEM_PROMPT_V3 to focus less on output formatting rules and more on what the output itself should contain (a 5-sentence answer that is concise and recommends the best 1 or 2 products and explains why), the LLM output became much tighter while still being grounded in the retrieved context. For example, V3 produced 2 product recommendations for the query "mechanical keyboard good for coding": [B0BGS7Q951] as the top pick, citing the 4.5-star rating, 104 keys, and wired USB suitable for PC, Mac, or laptop, and [B0016H8YW2] as a second good option while also citing the specific review snippet that mentioned the user had owned the keyboard since 1999 and it still worked perfectly, and that it had a compact size for carrying. The Product ASIN citations showed up in every sentence and the output stayed within the 5-sentence limit we set in the system prompt without ignoring specific product context details like ratings and review snippets. Since V3 does not hallucinate, cites every product it references, and explains its reasoning with actual review content from the context, while also being concise, we chose System Prompt 3 for our LLM based on the requirement that output must be concise. 

However, V3 has issues due to being too concise, which results in a less thorough comparison of products compared to V2. For example, V3 skipped the motospeed keyboard at rank 2 and jumped from rank 1 to rank 4 (happy hacking keyboard) for its second product recommendation, which makes V3's ranking aware recomemndation system worse compared to V2's outputs that typically follow ranking behavior better from the top k retrieved products. 

The 5-sentence limit in V3 also means the LLM is not able to discuss tradeoffs between more than 2 products in detail, which means V3 is not as useful for queries that ask to compare multiple products, even though it works well for "give me the best product" style queries. Lastly, our instruction on having an ASIN citation or product information per sentence in the output sometimes forces the LLM to output awkward sentence structures that do contain citations, but do not flow naturally.

Overall, V3 is better for when we want to output short answer recommendation queries where the user wants 1-2 top products with reasoning, while V2 is better  for broader comparison questions where the user wants to see more of the chain of thought reasoning steps of the LLM to see why specific products were recommended. V3 is our default prompt for the RAG pipeline since it provides a concise, more grounded summary that provides the user the ability to easily scan the output to see top products.

### Qualitative Evaluation for Hybrid RAG Workflow

| Query ID | Query | Difficulty | Accuracy | Completeness | Fluency | Key Observations |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | wireless bluetooth headphones | Easy | Yes | Yes | Yes | High accuracy; it correctly mapped ASIN `B0BM9JZDCH` features (bass, mic, noise-canceling) without hallucinations. |
| **2** | stainless steel water bottle 1 liter | Easy | Yes | Yes | Yes | Strong reasoning; it correctly identified the top result as a sleeve rather than the bottle itself, and noted the lack of a 1L bottle match. |
| **11** | headphones that don’t last long on a charge | Medium | Yes | No | Yes | Accuracy is good because it found a model that doesn't last long on charge, but completeness is lacking as the LLM also suggested "good" battery models, ignoring the negative intent. |
| **24** | office chair for back pain and long hours sitting | Difficult | No | No | Yes | It recommended a standing desk (ASIN `B07M9SFGBM`) instead of a chair, meaning that it misidentified the product category entirely. |
| **29** | monitor for coding with high resolution and low eye strain | Difficult | Yes | Yes | No | The response became highly repetitive, recommending the same product twice and including meta-talk about "strictly adhering to instructions." |

#### LLM Responses

##### Query: wireless bluetooth headphones (Success)
> I recommend [B0BM9JZDCH] for its wireless Bluetooth capabilities and features like deep bass built-in mic noise canceling, as well as compatibility with various devices including smartphones (ASIN B0BM9JZDCH). The second recommendation is [B08YVJN3GQ], which offers a durable hard case replacement specifically designed to fit Ultimate Ears UE true wireless Bluetooth earbuds and includes features like slip-proof design for easy carrying. Both products are highly rated, with an average rating of 4 or above (ASIN B0BM9JCH has no specified rating in the context).

##### Query: stainless steel water bottle 1 liter (Refusal Success)
> Based on your requirements for a stainless steel water bottle with at least one liter capacity, I recommend [B079L62G2J]. This product is specifically designed as an anti-slip silicone sleeve bottom protector that fits over hydro flasks and other bottles. It ensures the safety of chips in 32oz/40oz containers while providing great support to prevent slipping.
> 
> While there isn't a direct match for your exact specifications (1 liter capacity) within this context, [B079L62G2J] is closely related as it protects stainless steel water bottle bottoms and fits well with hydro flasks. If you need an unprotected 1-liter stainless steel water bottle itself without the silicone sleeve protector mentioned in these contexts, unfortunately there isn't enough information to recommend a product that meets your exact needs from this list of products provided.

##### Query: headphones that don’t last long on a charge (Constraint Failure)
> I recommend [B07TN3DVVL] for its 5 hours battery life which is longer than many other options listed here.
> 
> [Product rank: 2], B09CGKFQX3, has issues with charging and shutting off during use. Therefore it does not meet the criteria of lasting long on a charge.
> [Product rank: 4], Sony mdr-nc40 headphones offer up to 40 hours battery life which is significantly longer than other options listed here.
> 
> Based solely on this information I would recommend [B07TN3DVVL] for its better balance between price and performance. The second recommendation, the Sony mdr-nc40 headphone (ASIN: B008H0GCBG), offers a much higher battery life but is significantly more expensive than JVC's offering.

##### Query: office chair for back pain and long hours sitting (Accuracy Failure)
> Based on your requirements, I recommend [B07M9SFGBM] (Stand steady tranzform - tranzendesk electric standing desk with shelf) as it allows you to switch from sit to stand almost instantly which can alleviate the discomfort caused by prolonged periods of sitting. Additionally, consider using a memory foam cushion like fellowes professional series back support for immediate relief while working at your current setup [B00P1JYPXI].

##### Query: monitor for coding with high resolution and low eye strain (Fluency Failure)
> Based on the provided context [ASIN: B089FPZG9J], I recommend this BenQ GW2283 monitor. It features dual HDMI connectivity suitable for media devices switching which could be beneficial if you use multiple monitors or want to connect different types of displays, such as a TV and computer screen simultaneously.
> 
> Additionally, it has adaptive brightness technology that can help reduce eye strain during long periods of coding [ASIN: B0C65J4WDP]. This Gigastone monitor also offers built-in speakers which could be useful if you prefer an all-in-one setup for your workspace. Both monitors have high resolutions and are designed to minimize the risk of eye fatigue, making them suitable choices given their contextually relevant features [ASIN: B00NF3EMNA], with a slightly higher average rating indicating good customer satisfaction.
> 
> However, since we need only recommend one product based on strict adherence to your instructions:
> 
> I would choose ASIN: B089FPZG9J for its combination of dual HDMI ports and adaptive brightness technology specifically mentioned in the context as being optimized for home & office use. This makes it a versatile choice that directly addresses both high resolution needs (1080p IPS display) [ASIN: B00NF3EMNA] with eye care features, which is essential when coding long hours to minimize strain on your eyes.
> 
> The final recommendation of ASIN: B089FPZG9J stands out due to its dual HDMI ports and adaptive brightness technology that are directly mentioned in the product context as being optimized for home & office use. This makes it a versatile choice suitable not only for high-resolution displays but also specifically designed with eye care features, which is essential when coding long hours [ASIN: B00NF3EMNA].

#### Limitations
Based on the qualitative evaluation results, two limitations of the hybrid RAG workflow are:

**Failure to handle negative intent and specific constraints:** The system struggled with queries requiring the retrieval of products based on negative characteristics. For instance, when asked for "headphones that don’t last long on a charge," the model successfully identified a relevant low-battery model but also included recommendations for headphones with long battery life, effectively ignoring the user's negative constraint.

**Product category misidentification on difficult queries:** The workflow demonstrated an accuracy failure when processing complex queries. For example, when searching for an "office chair for back pain," the system retrieved and recommended an electric standing desk (ASIN B07M9SFGBM) instead of a chair, indicating that semantic similarity can lead to misidentifying the fundamental product category.



