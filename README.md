---
title: LitReviewAI
emoji: 💻
colorFrom: green
colorTo: pink
sdk: streamlit
sdk_version: 1.60.0
app_file: app.py
pinned: false
short_description: 👉 “Turn PDFs into Research Insights in Seconds."
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

```markdown
# 📚 LitReviewAI
## AI-Powered Research Literature Assistant


<p align="center">

<img src="assets/logo.png" width="180">

</p>


LitReviewAI is an intelligent research assistant designed to help researchers analyze, summarize, and organize scientific literature using Artificial Intelligence and Natural Language Processing.

It automates the tedious parts of literature review by extracting insights from research papers, identifying research gaps, discovering topics, and enabling interactive conversations with scientific documents.


---

# 🚀 Demo

🔗 Hugging Face Space:

(Add your Space link here)


---

# ✨ Features


## 📄 Automated Paper Analysis

Upload multiple research papers in PDF format and automatically extract:

- Paper title
- Authors
- Abstract
- Research information


---

## 🧠 AI-Powered Summarization

Using Large Language Models, LitReviewAI generates:

- Concise paper summaries
- Section-wise highlights
- Research limitations
- Potential research gaps


---

## 🔑 Keyword Extraction

Extracts important scientific keywords using:

- Sentence Transformers
- KeyBERT
- Semantic embeddings


---

## 📊 Topic Discovery

Discover research themes across multiple papers using:

- Latent Dirichlet Allocation (LDA)
- NLP-based topic modeling


---

## 🌐 Research Collaboration Network

Visualize:

- Author relationships
- Collaboration patterns
- Research communities


---

## 💬 Chat With Research Papers

Ask questions about uploaded papers and receive AI-generated answers based on extracted scientific content.


---

## 📚 Citation Management

Export analyzed papers into:

- BibTeX format
- CSV reports
- JSON summaries


---

# 🏗️ System Architecture



PDF Papers
|
|
PyMuPDF Extraction
|
|
Metadata + Abstract Extraction
|
|
AI/NLP Pipeline
|
|------------------
| |
Summarization Keyword Extraction
(Groq LLM) (KeyBERT)
|
|
Research Insights
|
|
Interactive Streamlit Dashboard



---

# 🛠️ Tech Stack


### Frontend

- Streamlit


### AI / NLP

- Groq LLM API
- Sentence Transformers
- KeyBERT
- Gensim LDA


### Document Processing

- PyMuPDF


### Visualization

- Plotly
- NetworkX
- WordCloud


### Data Processing

- Pandas
- Numpy


---

# 📂 Project Structure

LitReviewAI/

│
├── app.py
├── requirements.txt
│
├── src/
│ ├── embeddings.py
│ ├── metadata_extractor.py
│ ├── pdf_parser.py
│ ├── ai_functions.py
│ ├── topic_modeling.py
│ ├── visualizations.py
│ └── chat.py
│
├── components/
│ ├── upload.py
│ ├── summaries.py
│ ├── insights.py
│
├── assets/
│ └── logo.png
│
└── data/sample_papers



---

# ⚙️ Installation


Clone repository:

```bash
git clone https://github.com/Bano733-code/LitReviewAI.git

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py
🔐 Environment Variables

Create Streamlit secrets:

.streamlit/secrets.toml

Add:

GROQ_API_KEY="your_api_key"
📖 Research Applications

LitReviewAI can support:

Biomedical literature review
Computational biology research
AI-assisted scientific discovery
Systematic review workflows
Research hypothesis generation
🔬 Future Improvements

Planned features:

Vector database integration
Semantic paper search
RAG-based document retrieval
Citation recommendation
Automatic systematic review generation
Multi-document knowledge graphs
👩‍💻 Author

Bano Rani

BS Bioinformatics Student

Research Interests:

AI for Bioinformatics
Computational Biology
Biomedical NLP
Precision Medicine
📜 License

MIT License


---

This README will make LitReviewAI look like a **real AI research product**, not just a Streamlit assignment. It highlights the parts professors usually care about:

- scientific motivation
- AI methodology
- architecture
- reproducibility
- future research potential
- technical depth