# Multi-Modal RAG Search Engine

**Search inside PDFs + Images + Screenshots + Handwritten Notes — 100% Free & Lightning Fast**

Live Demo → [https://multimodelrag2.streamlit.app/](https://multimodelrag2.streamlit.app/)
GitHub → [https://github.com/ITianssj/Full-Stack-RAG](https://github.com/ITianssj/Multi-Modal-RAG-Search-Engine/tree/main)

---

### What It Does

Upload any document — PDF, image, or text file — and ask questions in plain English:

- “What does the diagram on page 3 show?”
- “Summarize the handwritten meeting notes”
- “Find all screenshots that mention ‘RAG’”
- “What are the numbers in the bar chart?”

The app instantly finds the answer **with sources**, even if the content is inside images.

---

### Features

| Feature                        | Implemented |
|-------------------------------|-------------|
| PDF + Image + TXT support     | Yes         |
| Vision understanding (charts, handwriting, diagrams) | Yes (Groq + Llama-3.2-Vision) |
| Semantic search across all content | Yes (Groq embeddings) |
| Real-time answers (<1 sec)    | Yes (Groq Llama-3.1-8B) |
| 100% Free forever             | Yes         |
| No local models (runs on phone-tier laptops) | Yes         |
| Clean database reset button   | Yes         |

---

### Tech Stack (All Free)

| Component         | Tool Used                            | Why |
|-------------------|--------------------------------------|-----|
| LLM               | Groq + Llama-3.1-8B-Instant          | 500+ tok/s, free API |
| Vision            | Groq + Llama-3.2-11B-Vision          | Best open-source vision model |
| Embeddings        | Groq + text-embedding-3-small        | Fast, free, no local download |
| Vector DB         | Chroma                               | Simple, persistent |
| UI & Deployment   | Streamlit                            | One-click deploy |
| OCR + Image Proc  | EasyOCR + pdfplumber                 | Handles handwriting + PDFs |

---

### How to Run Locally

```bash
# 1. Clone
git clone [https://github.com/ITianssj/Full-Stack-RAG.git](https://github.com/ITianssj/Multi-Modal-RAG-Search-Engine/tree/main)
cd /Multi-Modal-RAG-Search-Engine

# 2. Install
pip install -r requirements.txt

# 3. Add your free Groq key
cp .env.example .env
# Edit .env → put your key

# 4. Run
streamlit run app.py
```

Made with Love by Sagar Patel
Connect with me:

LinkedIn → https://www.linkedin.com/in/sagar-patel3996/
Medium → https://medium.com/@patel.sagar939
GitHub → https://github.com/ITianssj
