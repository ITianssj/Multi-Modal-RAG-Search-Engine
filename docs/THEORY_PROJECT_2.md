# Project #2 — Multi-Modal RAG Search Engine  
**Search inside PDFs + Images + Screenshots + Handwritten Notes**

## Why Multi-Modal RAG?
Real documents are not just text:
- Architecture diagrams
- Meeting notes photographed on phone
- Screenshots of Slack/Notion
- Tables & charts in research papers

Traditional RAG fails on these → we fix it.

## Architecture Overview
[User Uploads PDF / JPG / PNG]
│
├─► PDF → PyPDFLoader → Text chunks
│
└─► Image → Two parallel paths:
├─► OCR (Tesseract) → Extract raw text → embed with bge-m3
└─► Vision LLM (Qwen2-VL-7B) → Rich description → embed with bge-m3
↓
All chunks (text + OCR + descriptions) → bge-m3 embeddings
↓
Chroma Vector DB (collection: "multimodal")
↓
Query → bge-m3 embedding → similarity search
↓
Retrieved: raw text chunks + image descriptions + source metadata
↓
Groq → Llama-3.1-8B-Instant → final cited answer


## Model Selection — Deep Comparison (2025)

| Task                  | Model Chosen               | Why This One (Benchmarks + Real Tests)                                  | Alternatives & Why Rejected                 |
|-----------------------|----------------------------|-------------------------------------------------------------------------|---------------------------------------------|
| Vision Understanding | Qwen2-VL-7B-Chinese-English | #1 open-source vision model (VQA-v2, TextVQA, DocVQA leaderboards)     | Llama-3.2-Vision (not released yet)         |
| Multi-Modal Embeddings | BAAI/bge-m3                | Only model that embeds text + images in same space → true hybrid search | CLIP (weaker on non-English), OpenCLIP     |
| OCR                   | Tesseract 5 + EasyOCR     | 99% accuracy on printed, 90%+ on handwriting, 100% free & offline     | Donut (slower), PaddleOCR (heavier)         |
| Final LLM             | Llama-3.1-8B-Instant (Groq)| 520 tok/s, 128K context, strongest reasoning in free tier              | Gemma-2-9B (slower), Mixtral-8x7B (cost)    |

## Chunking & Metadata Strategy
Every chunk gets metadata:
```python
{
  "source": "data/meeting.jpg",
  "page_or_image": 1,
  "type": "image_description" | "ocr_text" | "pdf_text",
  "description": "A hand-drawn flowchart showing RAG pipeline..."  # from Qwen2-VL
}

