# Exploring Growth and Concentration in AI Research

COM4010 – Data Visualization Final Project
---

## 1. Project Overview

This project explores how AI research has evolved during the rapid expansion of modern artificial intelligence, particularly after the rise of deep learning, generative AI, and large language models.

Rather than focusing only on publication volume, the project investigates how scientific attention, citation impact, topic concentration, and institutional influence change over time.

The final product will be an interactive **Python Shiny dashboard** that guides users through four connected analytical views:

- Publication Growth
- Research Impact
- Concentration
- Research Pressure Indicators

### Main Research Question

> Is the rapid growth of AI research associated with broader scientific diversity and influence, or is attention becoming increasingly concentrated around a small number of topics, papers, and institutions?

---

## 2. Motivation

AI research is expanding too fast for researchers to track manually. This project is motivated by the need to provide a comprehensive, interactive map of the AI ecosystem, helping users identify not just how much AI research is growing, but who the hot spots are, who is leading them, and which domains are gaining traction

This project is motivated by the need to move beyond simple publication counts and examine how scientific attention and influence are distributed within the AI research ecosystem. The dashboard focuses on identifying patterns such as citation concentration, topic crowding, institutional dominance, and shifts in research attention over time.

---

## 3. Dataset Description

The primary dataset is collected from the **OpenAlex API**, an open scholarly database containing metadata on: Academic publications, Citations, Authors, Institutions, Venues, Research topics. 
The current sample dataset contains approximately: ~4,000 AI-related papers and ~ publication years: 2015–2026 .
The dataset is planned to expand to approximately: - 20,000–50,000 papers. 
The searching using broader AI-related search terms such as: - Machine Learning - Deep Learning - Large Language Models- Generative AI. 

### Dataset Features

| Category | Features | Description |
|---|---|---|
| Publication Metadata | Paper title, Publication year, Publication type, Venue/Source | Basic publication information describing when and where a paper was published |
| Impact Metrics | Citation count, Referenced works | Metrics related to research impact and citation relationships |
| Research Context | Topics, Keywords, AI subfields | Information describing the research domain and thematic focus of papers |
| Contributor Metadata | Authors, Institutions, Countries | Metadata about researchers, affiliations, and geographic distribution |


## Additional Data Sources

### Epoch AI Models Database
- Model scaling information
- Organization metadata
- Compute-related metrics

### Stanford AI Index
- Industry vs academia trends
- AI investment context
- Frontier model development statistics

---

## 4. Visualization Challenges

This dataset is non-trivial to visualize because it contains multiple interconnected dimensions, including:

- Publication growth over time
- Highly skewed citation distributions
- Overlapping research topics
- Institutional concentration
- Textual similarity across papers

The key design challenge is to avoid creating a static bibliometric dashboard.

Instead, the dashboard is designed to support a visual reasoning workflow where users:

1. Observe publication growth
2. Evaluate whether impact keeps up
3. Explore concentration patterns
4. Analyze combined research pressure indicators

This requires:
- Linked filtering
- Drill-down exploration
- Normalized metrics
- Interactive analytical components

---

## 5. Planned Dashboard Structure

### 1. Growth
Analyze publication expansion and topic growth over time.

### 2. Impact
Explore citation distributions, low-citation share, and research influence.

### 3. Concentration
Visualize concentration across countries, institutions, and research topics.

### 4. Pressure Indicators
Combine multiple analytical signals into interpretable indicators of research pressure and concentration.

---

## 7. Team Members & Responsibilities

## 👥 Team Members & Responsibilities

| Student ID | Team Member | Role | Responsibilities |
|---|---|---|---|
| V202401781 | Nguyen Thi Phuong Thao | Data Collection & EDA | OpenAlex API collection, data preprocessing, cleaning, exploratory data analysis (EDA), and dataset management |
| V202401694 | Le Thao Vy | Machine Learning & NLP Analysis | Topic modeling, clustering, textual similarity analysis, novelty exploration, and ML-based analytical methods |
| V202401748 | Ngo Thanh An | Visualization Development | Interactive charts, dashboard visualizations, map visualizations, and Python Shiny implementation |
| V2024 | Nguyen Hoang Nam | Storytelling & Dashboard Design | Dashboard structure, narrative flow, UI/UX consistency, presentation materials, and storytelling integration |
---

## 8. Project Status

### Current Progress
- Proposal completed
- Initial OpenAlex dataset collected
- Dashboard wireframe designed
- Preliminary EDA in progress

### Future Development
- Expand dataset size
- Build interactive dashboard
- Implement NLP-based similarity analysis
- Develop research pressure indicators
