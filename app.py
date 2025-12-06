"""
Multi-Modal RAG Search Engine - Streamlit UI

A production-ready Streamlit application for semantic search across PDFs, images, 
and text files using HuggingFace embeddings and Groq LLM.

Features:
- Upload & index PDFs, images, and text files
- Semantic similarity search
- Context-aware answers using Groq
- Chat history persistence
- Database management

Author: Sagar Patel
"""

import streamlit as st
import os
import shutil
import time
from groq import Groq
from config import settings
from ingest_multimodal import ingest_file, db
from loguru import logger

st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide"
)

st.title(settings.app_title)
st.caption("Search inside PDFs + Images + Screenshots | Groq + Chroma + Streamlit")

client = Groq(api_key=settings.groq_api_key)


# ============================================================================
# SIDEBAR: Document Management
# ============================================================================
with st.sidebar:
    st.header("📥 Upload & Index")
    st.markdown("Upload documents to build your searchable knowledge base.")
    
    upload = st.file_uploader(
        "Select a file",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        help="Supported: PDF, PNG, JPG, JPEG, TXT"
    )
    
    if upload and st.button("Ingest"):
        path = os.path.join(settings.data_folder, upload.name)
        os.makedirs(settings.data_folder, exist_ok=True)
        with open(path, "wb") as f:
            f.write(upload.getvalue())
        with st.spinner("Indexing..."):
            try:
                ingest_file(path)
                st.success("✅ Done! Now ask questions👇")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
                logger.error(f"Ingestion error: {e}")
    
    st.divider()
    
    if st.button("🗑️ Clear Database", help="Remove all indexed documents"):
        try:
            if hasattr(db, '_client'):
                db._client.close() if hasattr(db._client, 'close') else None
            
            if os.path.exists(settings.chroma_path):
                shutil.rmtree(settings.chroma_path, ignore_errors=True)
            
            time.sleep(0.5)
            
            st.success("✅ Database cleared. Reloading...")
            st.session_state.messages = []
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to clear database: {e}")
            logger.error(f"Clear DB error: {e}")


# ============================================================================
# CHAT INTERFACE
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# Chat input
if query := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching context..."):
            # Validate query
            if not isinstance(query, str):
                query = str(query)
            query = query.strip()
            
            if not query:
                st.error("Please enter a non-empty question.")
                docs = []
            else:
                try:
                    docs = db.similarity_search(query, k=settings.top_k)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    logger.error(f"Similarity search error: {e}")
                    docs = []
            
            context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."

            try:
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    temperature=0.1,
                    max_tokens=800,
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer ONLY using the provided context. If context is insufficient, say so."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query}"
                        }
                    ],
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = str(e)
                if "decommissioned" in error_msg or "deprecated" in error_msg:
                    st.error(
                        f"❌ Model Error: The configured model is no longer supported.\n\n"
                        f"**Recommended:** Update `config.py` to use `llama-3.1-70b-versatile` or "
                        f"`llama-3.1-8b-instant`.\n\n"
                        f"See: https://console.groq.com/docs/deprecations"
                    )
                else:
                    st.error(f"❌ LLM failed: {e}")
                logger.error(f"LLM error: {e}")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""Made with ❤️ by **Sagar Patel**  
Connect with me:  
- [LinkedIn](https://www.linkedin.com/in/sagar-patel3996/)  
- [Medium](https://medium.com/@patel.sagar939)  
- [GitHub](https://github.com/ITianssj)
""")
