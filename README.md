# LitReviewAI: Automated Research Paper Reviewer 📚🤖  

**Smarter Reviews, Faster Insights.**  
LitReviewAI is an AI-powered assistant that automatically reviews research papers, providing summaries, strengths, weaknesses, and improvement suggestions.  

---

## 🚀 Features
- 📄 Upload research papers (PDFs)  
- 🔎 Automatic summarization of key contributions  
- ✅ Strengths and weaknesses analysis  
- 💡 Suggestions for improvement & future directions  
- 🌍 Multilingual support (English + translations via Deep Translator)  
- 🔑 Keyword and topic extraction (KeyBERT, Gensim LDA)  
- 📊 Visualizations (WordClouds, Topic Graphs)  

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Frameworks**: Streamlit, Hugging Face Transformers, KeyBERT, Gensim  
- **Libraries**: PyMuPDF (fitz), Matplotlib, WordCloud, Deep Translator  
- **APIs**: Hugging Face Inference API, Deep Translator API  
- **Platforms**: Hugging Face Spaces / Streamlit Cloud  

---

## ⚙️ Installation (Local Setup)
```bash
# Clone repo
git clone https://github.com/Bano733-code/LitReviewAI.git
cd LitReviewAI

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```
▶️ Usage
Upload a research paper (PDF).
Choose Review Options: summarization, critique, keywords.
Get an AI-powered review report in seconds.
(Optional) Translate content into another language.

📈 Roadmap
Add plagiarism detection
Google Scholar/ArXiv API integration for related work
Dashboard for batch paper reviews
Fine-tuning on peer-review datasets

🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss what you’d like to change.

📜 License
MIT License

👩‍💻 Authors
Bano Rani – https://github.com/Bano733-code/LitReviewAI
