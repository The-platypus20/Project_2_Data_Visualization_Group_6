# Preprocess quality report

Input rows read: 2,239,384
Clean work rows: 2,239,384
Year range kept: 2000-2025
Years actually present: 2000-2025
Duplicate rows removed: 0
Rows dropped (bad/out-of-range year): 0
Rows dropped (missing primary_topic): 0
Missing title share: 0.05%
Missing country share: 29.60%
Missing venue_type share: 27.24%
Open-access share: 47.69%

## Citation distribution (clean rows)
- citation_count: min 0, median 1, mean 14.01, max 124822
- never cited (citation_count == 0): 38.53%
- citation_velocity: median 0.200, mean 1.440, max 18980.50

## Cleaning decisions
- Dropped rows with missing or out-of-range publication_year.
- Dropped rows with missing primary_topic.
- Deduplicated by paper_id first, then by title+year+primary_topic when id is unavailable.
- Filled numeric impact fields with 0 when missing (citation_count, citations_per_year, fwci, referenced_works_count, paper_age, citation_velocity, author_count, institution_count, country_count).
- Kept all original columns so downstream cache builders are unchanged.

## Column dtypes (as read)
- paper_id: object
- title: object
- publication_year: object
- publication_date: object
- publication_type: object
- citation_count: object
- citations_per_year: object
- fwci: object
- referenced_works_count: object
- topics: object
- keywords: object
- primary_topic: object
- primary_subfield: object
- primary_field: object
- primary_domain: object
- venue_source: object
- venue_type: object
- authors: object
- institutions: object
- institution_ids: object
- countries: object
- country_names: object
- doi: object
- language: object
- is_oa: object
- oa_status: object
- search_term_used: object
- paper_age: object
- citation_velocity: object
- author_count: object
- institution_count: object
- country_count: object

## Top 15 primary topics
- Natural Language Processing Techniques: 119,604
- Topic Modeling: 107,321
- Neural Networks and Applications: 105,167
- Quantum Information and Cryptography: 82,033
- Semantic Web and Ontologies: 74,773
- Quantum Computing Algorithms and Architecture: 62,776
- Geochemistry and Geologic Mapping: 58,254
- Privacy-Preserving Technologies in Data: 56,269
- Cryptography and Data Security: 53,407
- Speech Recognition and Synthesis: 52,506
- Anomaly Detection Techniques and Applications: 52,425
- Educational Robotics and Engineering: 50,483
- AI in cancer detection: 48,447
- Metaheuristic Optimization Algorithms Research: 47,182
- Edcuational Technology Systems: 43,050

## Top 15 primary countries
- China: 311,452
- United States: 257,842
- India: 96,059
- Germany: 70,278
- United Kingdom: 66,419
- Japan: 63,600
- Indonesia: 55,603
- France: 47,788
- Italy: 44,379
- Canada: 38,374
- Spain: 32,913
- Australia: 30,752
- South Korea: 26,602
- Brazil: 26,591
- Russia: 24,018

## Venue type distribution
- journal: 926,985
- (unknown): 609,995
- repository: 361,643
- book series: 218,112
- conference: 64,410
- ebook platform: 58,235
- other: 4

## Dashboard handoff
Use `Dataset/clean/ai_works_clean.csv` as the single `--input` to every build_*_cache.py script.
Cache CSV structure is identical to before; only counts reflect the cleaned data.
