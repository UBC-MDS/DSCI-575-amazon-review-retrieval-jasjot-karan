# Milestone 1 Discussion: Qualitative Evaluation of Retrieval Methods

## 4. Qualitative Evaluation

### 4.1 Query Set
The following query set was curated to evaluate the retrieval performance of the system across a spectrum of linguistic complexity. It consists of **30 unique queries** categorized into three difficulty levels: **Easy**, **Medium**, and **Difficult**. This balanced distribution allows for a granular assessment of how the retrieval engine handles straightforward keyword matching, broader semantic intent, and complex multi-constraint requests.

| Query | Difficulty |
| :--- | :--- |
| wireless bluetooth headphones | easy |
| stainless steel water bottle 1 liter | easy |
| gaming mechanical keyboard rgb | easy |
| noise cancelling earbuds | easy |
| office chair ergonomic | easy |
| laptop backpack waterproof | easy |
| earbuds secure fit running | easy |
| speaker loud bass | easy |
| 4k monitor 27 inch | easy |
| iphone fast charger cable | easy |
| headphones that don’t last long on a charge | medium |
| something that keeps water cold all day | medium |
| chair that hurts your back after sitting | medium |
| keyboard that is quiet when typing | medium |
| earbuds that won’t fall out during workouts | medium |
| laptop gets too hot when running programs | medium |
| backpack that breaks easily over time | medium |
| speaker not loud enough for a room | medium |
| monitor that is easy on the eyes for long use | medium |
| charger cable that stops working quickly | medium |
| best headphones for long flights under 200 | difficult |
| water bottle for hiking that stays cold in hot weather | difficult |
| good laptop for data science with long battery life | difficult |
| office chair for back pain and long hours sitting | difficult |
| earbuds for workouts that stay in and sound good | difficult |
| lightweight laptop for travel with good battery life | difficult |
| durable backpack with laptop protection anti theft | difficult |
| outdoor speaker with strong bass and waterproof | difficult |
| monitor for coding with high resolution and low eye strain | difficult |
| fast charging cable that is durable and long lasting | difficult |


### 4.2/4.3 Retrieval Results & Comparison
The following table compares the top 5 retrieval results using **BM25** (keyword-based) and **Semantic Search** (embedding-based).

| Query | BM25 Top Result | Semantic Top Result | Qualitative Observations |
| :--- | :--- | :--- | :--- |
| **wireless bluetooth headphones** | `co2crea hard case replacement for ultimate ears...` | `6s wireless bluetooth headphones over ear...` | **BM25 Failure:** Matched all keywords but returned an accessory (case). Semantic search correctly prioritized actual headphones. |
| **noise cancelling earbuds** | `co2crea hard case replacement for ultimate ears...` | `noise canceling stereo earphones. main category: all electronics...` | **BM25 Failure:** The keyword-based approach yielded a false positive, and the semantic search more accurately recognized the actual product category requested. |
| **laptop backpack waterproof** | `kopack laptop backpack, 17 in waterproof zipper...` | `aqua quest monsoon laptop case - 100% waterproof pouch...` | **BM25 Success:** Exact keyword matching worked for the product type "backpack". Semantic search was relevant but returned a "case/pouch" instead. |
| **speaker loud bass** | `soundboks (gen.3), green - loudest portable bluetooth...` | `altec lansing oem 3pc black pc music (221w)...` | **BM25 Success:** Correctly identified a high-powered speaker. Semantic results were relevant but less focused on the "loud/bass" intent. |
| **4k monitor 27 inch** | `t-power 19v ac dc adapter for phillips 22" 24" 27" monitors...` | `crua 27 inch 4k (3840x2160) monitor ips...` | **BM25 Failure:** Matched technical specifications in a power adapter title. Semantic search successfully retrieved the actual monitor. |
| **iphone fast charger cable** | `iphone charger 20ft/6m [apple mfi certified] lightning cable...` | `iphone 4 charger cables (3 pack 3.3 ft) 30 pin to usb...` | **Semantic Failure:** Returned obsolete 30-pin cables. BM25 was more helpful by matching "fast" and "lightning" keywords for modern devices. |
| **something that keeps water cold all day** | `stainless steel travel tumbler cup- -30 oz double wall vacuum...` | `homedics mychill personal space cooler... add water...` | **Semantic Failure:** The model "hallucinated" a personal fan because of the words "cold" and "water". BM25 correctly found a vacuum-insulated container. |
| **water bottle for hiking that stays cold in hot weather** | `thermos leak-proof commuter bottle - 16 oz. - charcoal...` | `sänger rubber hot water bottle - 2 litres...` | **Complex Query Failure:** Semantic search retrieved a "hot water bottle" (medical heat pack), failing the "cold" intent entirely. BM25 performed significantly better. |
| **good laptop for data science with long battery life** | `imation usb atom flash drive 8gb (66000105826)...` | `sony vaio vpcsa43fx/bi 13.3 inch laptop (jet black)...` | **BM25 Failure:** Returned a flash drive. Semantic search correctly identified a "laptop" even though the model is outdated, showing better category understanding. |
| **monitor for coding with high resolution and low eye strain** | `benq gw2283 eye care 22 inch ips 1080p monitor...` | `benq 24in led ips 1920x1200 16:10 aj bl2411pt...` | **Complex Query:** Both methods provided useful results, matching the "eye care/strain" and "resolution" constraints effectively. |

---

### Summary of Observations

The comparative performance of BM25 and Semantic Search was highly dependent on term specificity and the linguistic nature of the query. BM25 consistently performed better for technical/factoid queries with exact measurements or brand names, while Semantic Search excelled in intent-based queries where natural language described a use case rather than a specific product name. 

BM25's primary failure mode was Keyword Over-indexing, where it frequently retrieved accessories like cases or adapters due to high token overlap in titles. Semantic search more accurately recognized the core product categories in these instances. On the other hand, Semantic search struggled with negation, such as "don't last long", and adjective polarity, occasionally confusing "cold water" bottles with "hot water" bottles. It also faced issues with temporal relevance by returning obsolete hardware for general charger queries.

While the results were highly useful for "easy" queries, utility decreased as complexity increased. Retrieval models often matched the general product type but ignored complex numerical or conditional constraints. BM25 remained reliable for keyword-based (easy) queries, whereas semantic search provided significant value for "medium" queries by finding synonyms like "back support" for the query term "hurts back". The observed degradation in precision for "difficult" queries highlights a critical need for for more advanced methods, such as reranking, in future development to address specific user constraints.
