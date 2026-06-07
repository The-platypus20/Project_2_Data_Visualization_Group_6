# Preprocess quality report

Input rows read: 2,239,384
Clean work rows: 2,229,249
Country-long rows: 2,661,275
Year range kept: 2000-2025
Duplicate rows removed: 10,135
Missing title share: 0.00%
Missing country share: 0.00%

## Cleaning decisions
- Dropped rows with missing or out-of-range year.
- Dropped missing primary_topic rows by default. Use --keep-unknown-topic to keep them.
- Deduplicated first by id, then by title_clean + year + primary_topic when id is unavailable.
- Filled numeric impact fields with 0 when missing.
- Inferred topic_bucket from topic text when missing.
- Parsed multi-country fields into a country-long table.

## Top topics
- Natural Language Processing Techniques: 118,993
- Topic Modeling: 106,460
- Neural Networks and Applications: 104,838
- Quantum Information and Cryptography: 81,618
- Semantic Web and Ontologies: 74,529
- Quantum Computing Algorithms and Architecture: 62,237
- Geochemistry and Geologic Mapping: 58,233
- Privacy-Preserving Technologies in Data: 55,981
- Cryptography and Data Security: 53,239
- Speech Recognition and Synthesis: 52,322
- Anomaly Detection Techniques and Applications: 52,258
- Educational Robotics and Engineering: 50,482
- AI in cancer detection: 48,329
- Metaheuristic Optimization Algorithms Research: 47,084
- Edcuational Technology Systems: 43,038

## Top countries
- <NA>: 657,451
- CN: 337,014
- United States: 326,907
- IN: 105,423
- GB: 102,400
- DE: 92,733
- JP: 74,912
- FR: 66,406
- ID: 57,440
- IT: 57,318
- CA: 57,269
- AU: 47,997
- ES: 44,502
- KR: 32,790
- BR: 30,998

## Dashboard handoff
Use the clean work table as input to build_core_dashboard_cache.py. Use the country-long table for top_countries.csv and country_topic_year.csv when supported.
For Tab 3 ML, use impact_score or high_impact_label as target candidates, and use topic family, collaboration counts, venue type, open access, reference count, country count, and paper age as model features.