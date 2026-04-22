# Final Discussion

## Step 1: Improve Your Workflow

### LLM Experiment

#### Models Used

-   **Model 1:** `microsoft/Phi-4-mini-instruct` (3.8B parameters) - via Ollama as `phi4-mini`
-   **Model 2:** `meta-llama/Llama-3.2-3B-Instruct` (3B parameters) - via Ollama as `llama3.2:3b`
-   **Model 3:** `Qwen/Qwen2.5-3B` (3B parameters) - via Ollama as `qwen2.5:3b`

------------------------------------------------------------------------

#### Prompt

All five queries were run using identical retrieved context (top 5 documents via `HybridRetriever`) and the same prompt template defined in `src/prompts.py` as `SYSTEM_PROMPT_V3`.

```python
SYSTEM_PROMPT_V3 = """You are a strict Amazon shopping recommender. Answer the user's question in a concise manner using only the Product Context below in a maximum of 5 sentences. After looking at each product's metadata, recommend the best 1 or 2 products and state why you recommended them to the user.
Mandatory rules:
- Use ONLY the Product Context. Any claim not directly supported by the context must be omitted.
- Cite the ASIN in square brackets for every product referenced, e.g. [B01ABC123].
- The Product Context is ordered by retrieval relevance, with [Product rank: 1] being the most relevant to the user's query and higher ranks being progressively less relevant. If you recommend products that are not ranked in the top 3 in your answer; use the product metadata to briefly explain why you chose a worse ranked product.
- Every sentence must include at least one ASIN citation. Sentences without citations are not allowed.
- If the context is insufficient, respond with exactly this sentence and nothing else: "I do not have enough information to answer that. Please try searching another product."
- Do not begin with affirmations, compliments, or filler phrases.
- Do not end with emojis, exclamations, or questions.
- Do not produce template-like or generic answers. Every sentence must contain specific, context-grounded detail.
- Do not speculate about products outside the context, even if asked.
- Explain why you made the final product recommendation and base this explanation on the product context, with specific citations wherever you can. Do not reference the instructions.
"""
```

------------------------------------------------------------------------

#### Outputs

| Query | Phi-4-mini-instruct | Llama-3.2-3B-Instruct | Qwen2.5-3B |
|------------------|------------------|------------------|------------------|
| charger cable that stops working quickly | Based on user reviews, I recommend [B07ZYLF9WY] for its high-quality materials (4.3 average rating) despite some users reporting it stopped charging after about 30 days; this could be a minor issue compared to the overall positive feedback.\n\nI would also consider recommending [ASIN: B08N5Y1Y4T], as it's rated at an impressive 3.9, though reviews mention inconsistent performance over time (stopping charge). The user satisfaction seems high despite these issues.\n\n\nThe first recommendation is preferred due to its higher average rating and the second for similar reasons but with a slightly lower score; both have been reported by users who experienced some degradation in functionality after extended use. | Based on your query for a charger cable that stops working quickly, I recommend [B08N5Y1Y4T] [iPhone Charger, 10 ft Lightning Cable 3pack Fast Charging Nylon Braided iPhone Charging Cord, Compatible with iPhone 12/11/11 Pro/XS Max/XR/X/8/7/6s Plus/SE/IPad, Red&Black (10ft)] and [B09WK5VRK1] [3pack 6.6ft Multi Charging Cable 3 in 1 Charge Multi USB Charger Cable Aluminum Nylon Universal Multiple Cord].\n\nI chose these products because they have received reviews stating that the charging cables stop working after a few weeks, which aligns with your query. The [B08N5Y1Y4T] product has an average rating of 3.9 and multiple reviewers mention that it stops charging after a short period. On the other hand, the [B09WK5VRK1] product has an average rating of 3.0 and one reviewer explicitly states that it stopped working after about a month.\n\nWhile neither of these products is ranked in the top 3, they are the most relevant to your query based on their reviews. | For a charger cable that stops working quickly, I recommend [B071D5JG6N] droid turbo 2 compatible 3.1amp rapid car charger dc power adapter with usb port micro USB with touch activated led light coiled cable for motorola droid turbo 2. Although it did not work as expected and stopped charging after a few weeks, it is specifically designed to work with the Motorola Droid Turbo 2, which aligns well with your needs.\n\nIf you prefer a more versatile option that can be used with various devices including iPhones, I recommend [B085TH22W5] kinps mfi certified (3ft/1m) usb c to lightning fast charging cable. While it has a lower average rating of 4.7 compared to the Motorola charger's 4.0, its high-speed data transfer and official MFi certification make it more reliable for your needs. |
| chair that hurts your back after sitting | I recommend [B08ZSLRF2H] for its green background screen attachment designed specifically as an adjustable backdrop, which can be used with chairs (light green). However, if you're looking for a chair solution instead of just the accessory itself due to discomfort from prolonged use on other products like this one or others mentioned in reviews that indicate back pain issues after sitting too long. In such cases, [B07M9SFGBM] might also be considered as it is an electric standing desk with shelf which encourages changing positions frequently and can alleviate some of the strain caused by extended periods of sitting on a chair.\n\nThe first product was chosen because it's directly related to your query about using something for chairs. The second recommendation comes from its potential benefit in reducing back pain due to prolonged seating, as indicated indirectly through reviews mentioning discomfort after long-term use with other products like this one or others that are not recommended here. | [Product rank: 1] [B08ZSLRF2H]\nThis product is not suitable for alleviating back pain caused by sitting in a chair, as it is specifically designed as a portable green screen backdrop and does not provide any lumbar support.\n\n[Product rank: 3] [B00P1JYPXI]\\nThis product, the Fellowes Professional Series Back Support Memory Foam Cushion, may be suitable for alleviating back pain caused by sitting in a chair. It is designed to promote good posture with mid-spinal support and lower lumbar support, which could help alleviate discomfort.\n\nI recommend [B00P1JYPXI] as it provides specific features that address the issue of back pain caused by sitting in a chair, such as its mid-spinal support and memory foam cushion. | Based on the provided product context, [B08ZSLRF2H] seems like the best fit for the user's query as they are looking for a green screen backdrop attachment suitable for their chair. The title and description indicate it is designed specifically for photo/video studios with a portable green screen backdrop that fits over chairs. \n\nHowever, if you need a product related to office furniture or ergonomics, [B07M9SFGBM] might be more relevant as it offers an electric standing desk which can help reduce back pain associated with sitting at a desk for long periods.\n\nGiven the user's query about a chair that hurts their back after sitting, I recommend [B07M9SFGBM]. This product is specifically designed to alleviate back pain by allowing users to switch between sitting and standing positions easily. It fits better into the context of office furniture and ergonomics compared to the green screen backdrop attachment. |
| speaker not loud enough for a room | Based on your requirement of needing an extremely loud speaker suitable for larger rooms and considering durability as well (since you mentioned it should be portable), I recommend [B00JOQIDD2]. This Skullcandy Air Raid Water-resistant Drop-proof Bluetooth Portable Speaker is designed to handle rough use, features a 10-hour battery life with its rechargeable lithium-ion polymer battery pack. It also has an IPX7 rating for water resistance and can produce sound up to 90 dB at maximum volume.\n\nWhile [B01MDUCIEB] from Acoustic Audio by Goldwood offers high-quality audio (4.3 average rating) but is primarily designed as a bookshelf speaker, it may not meet your need for portability or the loudness required in larger rooms compared with Skullcandy's portable design and robust build that can handle rough use.\n\nTherefore, [B00JOQIDD2] would be better suited to provide both durability during transport (as you mentioned it's water-resistant) as well as sufficient volume output. | For a speaker that is not loud enough for a room, I recommend [B00JOQIDD2] [ASIN: B00JOQIDD2], the Skullcandy Air Raid Water-Resistant Drop-Proof Bluetooth Portable Speaker. This product has an average rating of 3.1 and features a rechargeable battery that lasts up to 10 hours, making it suitable for larger rooms.\n\nI chose this product over [B01MDUCIEB] [ASIN: B01MDUCIEB], the Acoustic Audio by Goldwood AA35CB Indoor Center 3 Way Speaker, because while it has a higher average rating (4.3), its power handling is only 400 watts (200w rms) which may not be sufficient for larger rooms.\n\nI did not recommend [B0178KMDEK] [ASIN: B0178KMDEK], the Cambridge Soundworks Oontz Angle 3, because while it has a high average rating (4.2), its battery life is only up to 7 hours which may not be enough for larger rooms.\n\nI did not recommend [B004N1QVAK] [ASIN: B004N1QVAK], the Advent AHT850 Center Channel Speaker, because its price and description are insufficient to determine its suitability for a room of that size. | For a speaker that is not loud enough for a room, I recommend [B01MDUCIEB] as it offers good sound quality and durability with a 4.3 average rating. Despite its lower price point compared to the first product, this speaker has received positive reviews from customers who appreciate its unique color, battery life, and overall performance in various settings.\n\nThe second recommendation is [B0178KMDEK], which offers excellent sound quality, water resistance, and a 4.2 average rating. Although it costs more than the first product, this speaker can handle up to 400 watts of power, making it suitable for use in home theaters or other loud environments where volume is crucial. |
| laptop gets too hot when running programs | I recommend [B08CVQKQBX] for its excellent performance and lack of overheating issues as described in user reviews; this laptop is praised highly with a rating of 5.0 stars by Computer Upgrade King, indicating it runs great without getting excessively warm even under heavy use.\n\nAdditionally, I would suggest avoiding the product at [B08J2M32VT] due to its poor performance and safety concerns related to overheating as mentioned in user reviews from eokoowo; this charger overheats significantly when used. | [Product rank: 1] [B013BYA46G]\nThis product is recommended for its ability to cool down a laptop that runs hot during intense usage. The reviewer mentions that the cooling pad fits perfectly and works well, making it suitable for laptops like the MacBook Pro that run hot when charging or using resource-intensive programs.\n\n[Product rank: 2] [B08CVQKQBX]\\nAlthough this product is not specifically designed as a cooling solution, its high-performance specifications (NVIDIA GeForce RTX 2060, Intel Core i7) and fast boot times suggest it may be less prone to overheating. However, the reviewer's glowing praise for the laptop's performance does not directly address heat issues.\n\n[Product rank: 3] [B004IMSSAU]\\nThis product is an older CPU that has been overclocked without issue by the reviewer, indicating it can handle demanding tasks without excessive heat generation. However, its low power consumption and outdated architecture make it less suitable for modern laptops.\n\nI recommend these products based on their relevance to cooling laptop overheating issues, with [B013BYA46G] being the top choice due to its explicit design as a cooling solution. | For a laptop that frequently overheats during intensive tasks like using Photoshop or running tech tools for repair, the best cooling solution is [B013BYA46G]. This cooling pad fits perfectly and effectively cools your 17-inch MacBook Pro. It has been specifically designed to handle high temperatures, which is crucial given the intense usage scenarios you mentioned.\n\nIf you are looking for a more comprehensive laptop upgrade that can also help with overheating issues, consider [B08CVQKQBX]. This professional notebook from MSI features advanced cooling systems and powerful hardware specifications, ensuring it remains cool even under heavy load. |
| speaker loud bass | Based on your interest in speakers with powerful, deep bass sound quality for parties and outdoor use [B07WSNJ2WF] from Soundboks would be the best recommendation due to its 126db volume level which is suitable for high-energy environments. Additionally, it offers features like Bluetooth connectivity (BT5.0), swappable battery providing up to 40 hours of playtime making it ideal for extended outdoor use [B07WSNJ2WF]. | Based on the Product Context, I recommend [B07WSNJ2WF] and [B00006HYI4]. \n\n[B07WSNJ2WF] is ranked #1 due to its high average rating of 4.4 and numerous reviews praising its loudness and deep bass performance. The product description also highlights its "big, powerful, deep bass" feature.\n\nOn the other hand, [B00006HYI4] is ranked #2 despite a lower average rating of 2.5, due to its high-wattage output (221w) which suggests it can produce loud sound. However, reviews indicate that this speaker excels in gaming rather than music or movie listening.\n\nI chose these two products over the others because they are most relevant to the user's query about "speaker loud bass". | For speaker loudness and bass, I recommend [B07WSNJ2WF] soundboks. It stands out with its impressive 126 dB of volume and deep bass capabilities, perfect for outdoor events like parties or bus rides. The user's positive experience in the review snippets indicates that this product delivers on these features effectively.\n\nIf you prefer a more portable option, I also recommend [B094Y1K678] Sony SRS-XB13. Although it has less powerful bass compared to soundboks, its 16-hour battery life and waterproof design make it an excellent choice for outdoor activities. |

------------------------------------------------------------------------

#### Key Observations

##### Response Style

Each model exhibited a distinct generation style that persisted across all five queries, suggesting these differences reflect fundamental tendencies of the model rather than query-specific variation.

- **`Phi-4-mini-instruct`** consistently produced conversational, review-grounded responses. Before making a recommendation, it typically restated the user's need and framed its reasoning in terms of product metadata, such as ratings, review sentiment, and feature descriptions. This makes its outputs feel the most natural and user-friendly of the three, closely resembling how a knowledgeable human shopper might explain a recommendation. However, this style occasionally led to verbose responses that exceeded the 5-sentence constraint specified in `SYSTEM_PROMPT_V3`.

- **`Llama-3.2-3B-Instruct`** adopted a more structured, enumeration-driven style. It frequently prefixed responses with explicit product rank labels (e.g. `[Product rank: 1]`) and followed a formulaic pattern: rank → ASIN → justification. While this makes the reasoning chain easy to follow, it can feel mechanical and less tailored to the user's actual question. This style also caused Llama to surface retrieval rank as a primary signal, even when higher-ranked products were not the best fit for the query.

- **`Qwen2.5-3B`** produced fluent, natural-language prose similar in tone to Phi-4, but with slightly less depth when referencing specific review details. Qwen tended to make confident product recommendations with clean sentence structure and good adherence to the citation rule. In comparison to Phi-4, it was more concise, although this sometimes came at the cost of justification detail.

---

##### Relevance and Query Interpretation

All three models successfully grounded their recommendations in the retrieved product context and cited ASINs in most responses. However, meaningful differences emerged when queries were ambiguous or when retrieved documents were remotely related to the user's query.

- **Straightforward queries** (e.g. *"speaker loud bass"*, *"charger cable that stops working quickly"*) produced strong alignment across all models. Each correctly identified the most relevant ASINs and referenced metadata, such as average ratings, review snippets, and feature descriptions to justify recommendations.

- **Ambiguous queries** exposed clear differences in interpretation strategy. For *"chair that hurts your back after sitting"*, the intent is likely to find an ergonomic or supportive chair — but the top-retrieved document (`[B08ZSLRF2H]`) was a green screen backdrop, not a chair at all. Phi-4 and Qwen both attempted to reinterpret the query constructively, pivoting to ergonomic alternatives like a standing desk (`[B07M9SFGBM]`) or back support cushion (`[B00P1JYPXI]`). Llama, by contrast, defaulted to the retrieval-ranked ordering and surfaced the green screen product first, only mentioning the back support as a secondary result, which is a less helpful response given the query's clear ergonomic intent.

- **Hallucination and over-reach:** Phi-4 occasionally made inferences that went slightly beyond what the product context supported. For example, in the *"speaker not loud enough for a room"* query, it referenced an IPX7 water-resistance rating and specific decibel figures that may not have appeared in the retrieved context verbatim. This suggests Phi-4 is more prone to supplementing retrieved evidence with parametric knowledge, which violates the `SYSTEM_PROMPT_V3` constraint to use *only* the Product Context.

---

##### Instruction Following and Consistency

Adherence to the mandatory rules in `SYSTEM_PROMPT_V3` varied noticeably across models, particularly for formatting constraints and citation requirements.

- **`Phi-4-mini-instruct`** showed strong overall instruction following, consistently citing ASINs and avoiding filler openers. Its main weakness was length control because several responses exceeded the 5-sentence limit, especially when it chose to recommend two products and justify both in detail. It also occasionally used raw newline escape sequences (`\n\nI would also...`) in its output rather than actual line breaks, indicating some inconsistency in how it handles formatting within the generation.

- **`Llama-3.2-3B-Instruct`** was the weakest in terms of clean formatting. It regularly emitted raw escape sequences (`\n`, `\\n`) as literal text within responses, and sometimes reproduced internal template-like labels (e.g. `[Product rank: 1]`, `[ASIN: B00...]`) that were likely artifacts of its instruction-tuning format leaking into the output. This suggests that at 3B scale, Llama has difficulty fully separating structured reasoning from free-form generation. Despite these artifacts, it was reasonably consistent in citing ASINs and respecting the sentence-count limit.

- **`Qwen2.5-3B`** was the most consistent model across all five queries. It reliably cited ASINs, stayed within the 5-sentence guideline, and produced clean output without formatting artifacts. Its instruction following was notably better than Llama's and more disciplined than Phi-4's in terms of response length, making it the most predictable model for production use under this prompt configuration.

---

##### Overall Model Ranking for This Task

Based on the observations above, the three models can be ranked for this specific RAG-based product recommendation task as follows:

| Rank | Model | Strengths | Weaknesses |
|------|-------|-----------|------------|
| 1 | `Qwen2.5-3B` | Clean formatting, consistent citation, concise and natural prose | Slightly less detail in justifications |
| 2 | `Phi-4-mini-instruct` | Most natural and user-friendly tone, strong reasoning depth | Verbose, minor hallucination risk, formatting inconsistencies |
| 3 | `Llama-3.2-3B-Instruct` | Transparent ranking logic, reliable ASIN citation | Mechanical style, formatting artifacts, weaker query interpretation |

All three models are competitive for a 3B parameter class, and the differences may narrow with prompt tuning or fine-tuning on domain-specific data. However, for out-of-the-box performance under `SYSTEM_PROMPT_V3`, Qwen2.5-3B offers the best balance of quality, consistency, and instruction adherence.

------------------------------------------------------------------------

## Step 2: Tool Integration Examples & Dataset Scaling

The web search tool was used on the 3 queries below and produced more relevant, current responses compared to running the pipeline without web search. The web search tool is triggered by an LLM-based router (phi4-mini) that evaluates each user query itself and decides whether live web data is needed. If the user query mentions current stock availability, prices, where to buy the item, or recent news and updates about the product, the router outputs "yes" and Tavily web search is called, appending the web search results to the local retrieval context before the final answer is generated. If the query is about product reviews, feature comparisons, or general recommendations, the router outputs "no" and only the local retrieval is used, which avoids unneeded web calls that would increase inference time for users.

### Example 1: Current Price and Availability

**Query:** *"What is the current price and stock availability of the Apple AirPods Pro 2 on Amazon and Best Buy as of today?"*

- **Without tool:** Response is based on the static 2023 Amazon product metadata and output says that "the current price is not provided in Amazon's Product Context and Best Buy stock availability cannot be determined as they are third-party retailers outside of this context". 
  information stale, no retailer comparison, no stock status.
- **With tool:** Response included current price of Airpods Pro 2 on Amazon: $349 and says that the price is around the same at Best Buy, however, the exact price today cannot be confirmed for sure. The output is not overconfident and recommends checking product: B08D7X494N (Airpods Pro 2) on Amazon, and to visit Apple's website to get officail product information and prices today. 
- **Conclusion with Web Search:** Clear improvement with web search. Static product metadata corpus cannot answer time-sensitive pricing questions and mentions it does not contain pricing or stock availability. Tavily fills this gap and mentions the price of the product and to check a specific product on Amazon, and also to visit Apple's website to get up to date stock availability information. 

### Example 2: Latest Product Version
**Query:** *"What is the newest version of a Keychron K3 mechanical keyboard and the current price nowadays?"*

- **Without tool:** The system retrieved a few related but not matching
  mechanical keyboards from the 2023 Amazon corpus (ASINs B07YJRWQDL, 
  B08YJ7VXYR, B07NQ2HXC5), none of which are actually Keychron K3 
  keyboards. The LLM honestly flagged this shortcoming and responded: "I do not have enough information to answer that. Please try searching another product."
- **With tool:** Tavily returned up-to-date information on the current 
  Keychron K3 keyboards that included the newest version (K3 Max / Pro 
  variants) and also returned pricing from the official Keychron site .
- **Conclusion with Web Search:** Clear improvement with web search. Our 2023 Amazon Electronics product metdata does not have the Keychron K3 in it, so local retrieval could only return related, older Keychron keyboards. This web search helped with answering queries about newer product generations of the Keychron keyboard series.

### Example 3: Where to Buy
**Query:** *"Where can I buy the Samsung Galaxy Buds 3 Pro in Canada right now, and what are the best deals available?"*

- **Without tool:** Local retrieval without web search returned mostly 
  products tbat were not relevant or related to the user query, such as  a Samsung earbud accessory/replacement part ([ASIN: B09M3TDJZ6]) from 
  iiexcel on Amazon for $9.99, and an an incompatible product ([ASIN: B09Q35KXVS]). The LLM did honestly mention that it could not actually answer the question and outputted that the retrieved products "do not match the Samsung Galaxy Buds 3 Pro" and that the local product corpus does not provide Canadian availability information.
- **With tool:** Tavily web search added Walmart.ca as a Canadian retailer that carries the earbuds, but the response was still not trustworthy as it did not include current pricing or stock availability in Canada from Walmart or any other site for the actual Galaxy Buds 3 Pro. The first product returned in the response with web search was also still the irrelevant $9.99 accessory (not the actual earbuds) that was retrieved directly from the local product metadata corpus.
- **Conclusion with Web Search:** Web search gave us a partial imporvement. Tavily added one Canadian retailer (Walmart) that the static corpus could not add since it is US based, but the overall response was still weak because the main product outputted as a recommendation was still from our local Amazon product metadata context and was not related to the user query about the actual product directly. Therefore, we saw that when local product retrieval from the static corpus returns low-quality matches, final output from the LLM can let that dominate and not include much of the web search's context.  

### Dataset Scaling 

#### Number of Products Used

The pipeline scales to and contains **200,000 products**, controlled by the constant variable: `MAX_PRODUCTS = 200_000` at the top of `notebooksmilestone1_exploration.ipynb`. 

#### Sampling Strategy and Engineering Decisions

The raw Amazon Electronics dataset is too big for a standard laptop, since `meta_Electronics.jsonl` has ~1.6M products and `Electronics.jsonl` has ~18.3M reviews. Loading both into memory all at once with pandas would cause out of memory (OOM) errors. To load all of the rows in each dataset for downstream use without cloud compute, we used the following engineering design decisions:

**Polars lazy evaluation with streaming:** Both files are opened with 
`pl.scan_ndjson()`, that build a lazy query plan only instead of loading the entire datasets into memory. The final corpus used for retrieval is written using `sink_parquet()`, which automatically streams results to 
the saved file on disk in chunks instead of holding the full dataset in RAM at once. This lazy loading with Polars allows the pipeline to be runnable on a standard laptop.

**Limiting sampling by `parent_asin`.** We select the first 200K distinct `parent_asin` (product ID) values from the product metadata file, then filter the 18.3M review dataset down using a semi-join on that ASIN set. This makes sure that we limit the number of reviews to only contain reviews of the sampled 200K products.

**Review aggregation.** We do not store every single review, as this would result in having multiple rows for each product, if each product has multiple reviews, which can result in duplicate output for our actual application. Therefore, we group the reviews by `parent_asin` and aggregate each product's review into two fields: `all_review_titles` (concatenated titles) and `review_text_200` (first 200 characters of combined review text).

**Left-join to keep products without reviews.** A `left_join` on `parent_asin` when joining the reviews table to the product metadata table guarantees that there are 200K rows in the final corpus, even if there are products with no matching reviews.

**Single `retrieval_text` field for retrieval.** We concatenate parent_asin, title, category, store, features, description, average rating, review titles, and the review snippet into one searchable field so downstream BM25 and FAISS retrievers can index one column per product, that contains a lot of rich metadata the user can ask about.

------------------------------------------------------------------------

## Step 3: Improve Documentation and Code Quality

### Documentation Update

The `README.md` file was updated to make the repository cleaner and more professional. Key improvements include:

- Added shields.io badges for Python version, Streamlit, FAISS, and MIT License.
- Added a Table of Contents with anchor links for easier navigation.
- Added a Usage Examples section with five distinct query types mapped to their expected best retrieval method.
- Updated the Project Structure section with an annotated file tree describing the purpose of each file and directory.

### Code Quality Changes

- Replaced all hardcoded string paths across `src/` with `pathlib.Path` objects and the `/` operator for path joining, making file paths OS-independent and more readable.
- Added a minimum one-line docstring to all functions across `src/` files (including `build_prompt` in `prompts.py` and `run_semantic_search` in `semantic.py`) to improve code readability and self-documentation.
- Added `.Rhistory` to `.gitignore` to prevent R session history files from being tracked in the repository.

## Step 4: Cloud Deployment Plan

Deploying our Amazon Product Retrieval app on AWS would require: 

### Data Storage 

Raw data files: `Electronics.jsonl` and `meta_Electronics.jsonl` will be stored in an Amazon S3 Bucket called: `s3://amazon-retrieval-raw/`. AWS S2 is a good storage method in this case since the raw review file: ``Electronics.jsonl` has 18.3 million reviews and `meta_Electronics.jsonl` has 1.6 million products, which are a lot of GBs and would take up a lot of disk space if stored locally. Amazon S3 allows you to store a lot of GBs of data at very low prices in a secure, and scalable environment that is fault tolerant. This means that S3 is designed to prevent against data loss, since it replicates your data across different data centers that are seperate from each other which makes sure that if one data center fails, the other zones continue to operate which keeps your data safe. 

After preprocessing and combining the data needed for retrieval, such as the product metadata along with selected review information, we can store the processed folder: `data/processed/retrieval_corpus.parquet` in Amazon S3 in a separate bucket. This bucket can have a different name/prefix to distinguish that it is processed data inttead of raw data such as: `s3://amazon-retrieval-processed/corpus/`. Our iniital data reading and proprocessing steps use Polars, which supports reading in parquet file formats and S3 bucket filepaths, which would allow us to read the processed data downstream directly from S3 using a line like: `pl.scan_parquet("s3://...")`. We can also add a timestamp postfix to the file: `retrieval_corpus.parquet` that is saved to S3 that allows us to have versioning and roll back to a previous version if a preprocessing step produces a bad corpus version we do not want. 

The FAISS vector index and the embedding chunk partitions: `embedding_chunks/*.npy` could be stored in Amazon S3 as an initial storage layer on the first run and to read from in a folder such as: `s3://amazon-retrieval-processed/faiss/`. At future app startups, the EC2 instance that we run the app on could download the existing FAISS vector index from `s3://amazon-retrieval-processed/faiss/` to its local volume so the existing index is stored inside the EC2 instances project directory's `processed/data` folder so it is automatically loaded on app startup for future runs. The BM25 index: `bm25_index.pkl` and the metadata row corpus: `metadata_rows.pkl` that BM25 searches for similarity in can be stored in S3 in a seperate subfolder such a `s3://amazon-retrieval-processed/bm25/`. Therefore, the BM25 index and metadata rows that it uses for similarity search can be pre-loaded into memory on the app startup from S3, so the BM25 index and metadata rows do not have to be built from scratch on app startup. 

### Compute 

Our app should be on an a separate instance from our local machine, on an EC2 instance, which lets us host an application on a virtual server. Since our data preprocessing step and FAISS index/embedding creation take a long time locally, we could use a powerful instance with a lot of computing power and CPUs, such as a `m6i.2xlarge`, which offers 8 CPU cores and 32GB of RAM, allowing us to have much faster processing speeds. We could also use a Docker image by packing our ennvironments and run commands into a Dockerfile. This would allow us to use an app such as AWS App Runner for deployment of our app, which has auto-scaling, so we can handle a lot of users as they cone in, without our app crashing or having slower performance and inference times. 

We can handle multiple users by implementing Auto Scaling in EC2. This auto scaling would handle concurrency and allow for horizontal scaling automatically depending on the number of user requests. This means that the EC2 instance would add or remove entire EC2 instances to match the volume of user requests that are coming in. Therefore, if there are lots of user requests coming in, auto scaliling will add entire new EC2 instances comoared to resizing existing EC2 instances. New instances come with a load balancer, and this load balancer distributes multiple incoming user requests to all avaiable instances. This allows our app to scale to many different users without forcing users to wait for a single instance to free up. 

We would handle LLM inference using a hosted model service such as AWS Bedrock instead of using a local model such as Phi-4 mini on Ollama. This is because Bedrock offers all major frontier LLM providers managed APIs to run, such as Claude, LLama for example, so we do not have to download the model we want to run locally onto our own local disk. Using a hosted model on AWS Bedrock allows us to have a managed API where we can call major LLM APIs while not having to manage any local infastructure ourselves such as having our own powerful GPU. AWS Bedrock also offers auto scaling so that we can hamdle multiple users API calls easily, without having to wait on one single LLM API call to be returned before processing the next user request.

### Streaming Updates

We would incorporate new products into production by running a recurring daily batch job that takes new product metadata and that products review data as it comes in and drops them into an S3 raw storage folder within `s3://amazon-retrieval-raw/`. We cold setup event notifications in S3 so that we are notified when new raw product metadata and review data is detected in `s3://amazon-retrieval-raw/products-to-add`. These event notifications that new products were added could trigger an AWS Lambda function or AWS Step functions for our multi-step case, where the full pipeline is orchestrated again after concatenating the new product and review metdata to the raw `Electronics.jsonl` and `meta_Electronics.jsonl` files. These AWS Step functions could then take the new raw product and review metadata and orchestrate the pipeline steps to run Polars preprocessing again to clean the new product metdata, before creating a new BM25 index based on that data, and a new FAISS index & embeddings, before updating the updated metadata artifacts to S3 with a timestamp. 

Our pipeline would stay up to date by using a service such as AWS Eventbridge that could trigger the rebuilding of the pipeline depending on how often new data comes in. This service could dynamically swap variables when it runs this pipeline when rebuilding the indexes and embeddings so that the most updated data with the new products is used to generate embeddings and indices. We could also store health statistics such as the last successful run, and the corpus size, along with our user feedback metadata in a service such as DynamoDB. This would allow user feedback and health statistics of the app for monitoring to be stored in the same instance/place, regardless of what instance the user gets assigned to when using the app. This would allow user and health montitoring feedback to be stored in one unified place instead of being stuck in separate EC2 instances' local disks. 
