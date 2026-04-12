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

| Query | BM25 Results | Semantic Results | Qualitative Observations |
| :--- | :--- | :--- | :--- |
| **wireless bluetooth headphones** | 1. `co2crea hard case for ue fits...`<br>2. `case for dewalt bluetooth...`<br>3. `antfire true wireless earbuds...`<br>4. `wireless earbuds,bluetooth...`<br>5. `mkay wireless headphones...` | 1. `6s wireless headphones over ear...`<br>2. `poyatu headphone earpads...`<br>3. `lemoos wireless earbuds...`<br>4. `wireless earbuds, bluetoth...`<br>5. `truly wireless earbuds...` | **BM25 Failure:** The keyword-based approach yielded a false positive, and the semantic search more accurately recognized the actual product category requested. |
| **gaming mechanical keyboard rgb** | 1. `supercolor pc gaming keyboard...`<br>2. `gaming keyboard, rainbow backlit...`<br>3. `gaming keyboard combo...`<br>4. `supercolor sk61 led wired...`<br>5. `rk royal kludge rk61 rgb...` | 1. `nacodex mechanical gaming...`<br>2. `supercolor pc gaming...`<br>3. `supercolor pc gaming sk61...`<br>4. `havit mechanical keyboard...`<br>5. `rk royal kludge rk61 rgb...` | **Success:** Both methods performed exceptionally well due to the presence of highly specific technical keywords in product titles. |
| **laptop backpack waterproof** | 1. `kopack laptop backpack...`<br>2. `sosoon laptop backpack...`<br>3. `monsac laptop backpack...`<br>4. `laptop backpack business...`<br>5. `shrradoo extra large laptop...` | 1. `aqua quest monsoon laptop case...`<br>2. `aqua quest monsoon laptop case...`<br>3. `sosoon laptop backpack...`<br>4. `monsac laptop backpack...`<br>5. `laptop backpack for women...` | **Semantic Failure:** The semantic search returned waterproof "cases" in the top spots rather than the requested "backpack." BM25 was more accurate here. |
| **noise cancelling earbuds** | 1. `co2crea hard case for ue fits...`<br>2. `case for dewalt bluetooth...`<br>3. `hard case for jbl reflect...`<br>4. `antfire true wireless earbuds...`<br>5. `wireless earbuds,bluetooth...` | 1. `noise canceling stereo earphones...`<br>2. `antfire true wireless earbuds...`<br>3. `wireless earbuds, bluetoth...`<br>4. `truly wireless earbuds...`<br>5. `case for noise-canceling...` | **BM25 Failure:** Keyword stuffing in accessory titles led to several false positives; semantic search prioritized the electronic device. |
| **office chair ergonomic** | 1. `ergonomic keyboard stand...`<br>2. `monitor stand riser...`<br>3. `lap desk memory foam...`<br>4. `laptop stand adjustable...`<br>5. `boon ergonomic laptop stand...` | 1. `laptop stand adjustable...`<br>2. `monitor stand riser...`<br>3. `monitor stand with drawer...`<br>4. `laptop stand for desk...`<br>5. `lap desk memory foam...` | **Common Failure:** Both methods failed to distinguish the core product (chair) from ergonomic accessories like stands and lap desks. |
| **keeps water cold all day** | 1. `stainless steel travel tumbler...` | 1. `homedics mychill personal space cooler...` | **Semantic Failure:** The model "hallucinated" a personal cooling fan because of the keywords "cold" and "water". |
| **chair hurts back after sitting** | 1. `blackovis treeline sitting tripod...` | 1. `fellowes professional back support...` | **Semantic Success:** Identified the need for "back support" even though the keyword in the query was "hurts". |
| **keyboard quiet when typing** | 1. `wireless bluetooth keyboard quiet...` | 1. `taeeiancd typewriter keyboard...` | **BM25 Success:** Effectively matched the "quiet" adjective directly in the title. |
| **earbuds won’t fall out** | 1. `yurbuds leap wireless sport...` | 1. `wired headphones, in ear magnetic...` | **Comparison:** BM25 found "sport" earbuds for stability. Semantic search focused on magnetic secure features. |
| **don’t last long on a charge** | 1. `jvc wireless earbuds 5h battery...` | 1. `bluetooth headphones 16h playtime...` | **Constraint Failure:** Both methods failed the negative intent and returned products with *long* battery life. |
| **flights under 200** | 1. `sony mdr-nc40 noise cancelling...` | 1. `sony stereo open-air mdr-ma900...` | **Semantic Failure:** Returned "open-air" headphones, which are the opposite of what is needed for a flight. |
| **hiking stays cold in hot weather** | 1. `thermos leak-proof commuter bottle...` | 1. `sänger rubber hot water bottle...` | **Semantic Failure:** Hallucinated a medical "hot water bottle" due to word proximity. |
| **laptop for data science** | 1. `imation usb atom flash drive...` | 1. `sony vaio vpcsa43fx/bi laptop...` | **BM25 Failure:** Matched technical terms to a flash drive. Semantic search correctly identified the laptop category. |
| **office chair for back pain** | 1. `stand steady tranzform standing desk...` | 1. `fellowes professional back support...` | **Success:** Semantic search accurately matched the "back pain" intent to a corrective orthopedic cushion. |
| **workouts stay in and sound good** | 1. `red bluetooth sports earbuds...` | 1. `soundcore by anker sport x10...` | **Success:** Both methods found "sport" designated earbuds that address the stability requirement. |

---

### Summary of Observations

The comparative performance of BM25 and Semantic Search was highly dependent on term specificity and the linguistic nature of the query. BM25 consistently performed better for technical/factoid queries with exact measurements or brand names, while Semantic Search excelled in intent-based queries where natural language described a use case rather than a specific product name. 

BM25's primary failure mode was Keyword Over-indexing, where it frequently retrieved accessories like cases or adapters due to high token overlap in titles. Semantic search more accurately recognized the core product categories in these instances. On the other hand, Semantic search struggled with negation, such as "don't last long", and adjective polarity, occasionally confusing "cold water" bottles with "hot water" bottles. It also faced issues with temporal relevance by returning obsolete hardware for general charger queries.

While the results were highly useful for "easy" queries, utility decreased as complexity increased. Retrieval models often matched the general product type but ignored complex numerical or conditional constraints. BM25 remained reliable for keyword-based (easy) queries, whereas semantic search provided significant value for "medium" queries by finding synonyms like "back support" for the query term "hurts back". The observed degradation in precision for "difficult" queries highlights a critical need for for more advanced methods, such as reranking, in future development to address specific user constraints.
