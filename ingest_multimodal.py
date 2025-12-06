"""
Document Ingestion Module

Handles extraction and processing of multiple document types:
- PDFs: Text extraction + embedded images
- Images: Direct vision processing
- Text files: Plain text ingestion

All content is chunked, embedded, and stored in Chroma vector database.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from vision_processor import describe_image
from config import settings
from loguru import logger
import os
import pdfplumber
import json


# Initialize embeddings (runs locally - no API calls)
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize Chroma vector database
db = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=embeddings,
    collection_name=settings.collection_name
)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)


def _safe_add_texts(texts, metadatas):
    """
    Safely validate and add texts to vector database.
    
    Performs robust type conversion and filtering:
    - Converts bytes/non-strings to strings
    - Strips whitespace and removes empty entries
    - On batch failure, attempts per-item validation
    - Keeps metadata aligned with valid texts
    
    Args:
        texts (list[str): Document chunks to embed
        metadatas (list[dict]): Metadata for each chunk (source, page, type)
    
    Returns:
        None (modifies database in-place)
    
    Example:
        >>> _safe_add_texts(["Hello world"], [{"source": "test.txt", "type": "text"}])
    """
    if not texts or not metadatas:
        logger.debug("No texts/metadatas provided to _safe_add_texts()")
        return

    # Normalize and validate entries
    normalized = []
    for idx, (t, m) in enumerate(zip(texts, metadatas)):
        if t is None:
            logger.debug(f"Dropping None text at index {idx}")
            continue

        # Handle bytes
        if isinstance(t, (bytes, bytearray)):
            try:
                t = t.decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug(f"Failed to decode bytes at index {idx}: {e}")
                t = str(t)

        # Handle non-strings
        if not isinstance(t, str):
            try:
                if isinstance(t, (dict, list)):
                    t = json.dumps(t, ensure_ascii=False)
                else:
                    t = str(t)
            except Exception as e:
                logger.debug(f"Failed to convert text to str at index {idx}: {e}")
                continue

        t = t.strip()
        if not t:
            logger.debug(f"Dropping empty text at index {idx}")
            continue

        normalized.append((t, m))

    if not normalized:
        logger.warning("No valid text entries after normalization")
        return

    final_texts, final_metas = zip(*normalized)
    final_texts = list(final_texts)
    final_metas = list(final_metas)

    # Attempt batch add
    try:
        db.add_texts(texts=final_texts, metadatas=final_metas)
        logger.debug(f"Added {len(final_texts)} texts to DB (batch)")
        return
    except Exception as e:
        logger.warning(f"Batch add failed: {e}. Attempting per-item validation...")

    # Fallback: validate per-item
    good_texts = []
    good_metas = []
    for i, (t, m) in enumerate(zip(final_texts, final_metas)):
        try:
            embeddings.embed_documents([t])
            good_texts.append(t)
            good_metas.append(m)
        except Exception:
            logger.debug(f"Dropping text at index {i} (embedding failed)")

    if not good_texts:
        logger.error("All items failed validation; nothing added to DB")
        return

    try:
        db.add_texts(texts=good_texts, metadatas=good_metas)
        logger.info(f"Added {len(good_texts)} texts (dropped {len(final_texts) - len(good_texts)})")
    except Exception as e:
        logger.exception(f"Final add failed: {e}")
        raise


def ingest_pdf(file_path: str):
    """
    Extract and ingest text and images from PDF.
    
    Args:
        file_path (str): Path to PDF file
    
    Returns:
        None (modifies database)
    """
    logger.info(f"Processing PDF: {file_path}")
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract text
            text = page.extract_text()
            if text and text.strip():
                chunks = text_splitter.split_text(text)
                chunks = [c for c in chunks if c.strip()]
                if chunks:
                    _safe_add_texts(
                        chunks,
                        [{"source": file_path, "page": page_num, "type": "pdf_text"}] * len(chunks)
                    )

            # Extract images - use page.images if available, otherwise use extract_images()
            try:
                image_list = page.extract_images() if hasattr(page, 'extract_images') else []
            except AttributeError:
                image_list = []
            
            # Fallback to page.images if extract_images() doesn't exist
            if not image_list and hasattr(page, 'images'):
                image_list = page.images
            
            for img_index, img in enumerate(image_list):
                try:
                    temp_path = f"temp_page_{page_num}_img_{img_index}.png"
                    
                    # Handle both dict (from extract_images) and Image object (from images)
                    if isinstance(img, dict) and "stream" in img:
                        with open(temp_path, "wb") as img_file:
                            img_file.write(img["stream"].get_data())
                    elif hasattr(img, 'stream'):
                        with open(temp_path, "wb") as img_file:
                            img_file.write(img.stream.get_data())
                    else:
                        logger.debug(f"Unknown image format on page {page_num}, skipping")
                        continue
                    
                    description = describe_image(temp_path)
                    if description:
                        _safe_add_texts(
                            [str(description).strip()],
                            [{"source": file_path, "page": page_num, "type": "image_description"}]
                        )
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to process image on page {page_num}: {e}")


def ingest_image(file_path: str):
    """
    Generate description for image and ingest.
    
    Args:
        file_path (str): Path to image file
    
    Returns:
        None (modifies database)
    """
    logger.info(f"Processing image: {file_path}")
    description = describe_image(file_path)
    if description:
        _safe_add_texts(
            [str(description).strip()],
            [{"source": file_path, "type": "image_description"}]
        )


def ingest_txt(file_path: str):
    """
    Load and ingest plain text file.
    
    Args:
        file_path (str): Path to text file
    
    Returns:
        None (modifies database)
    """
    logger.info(f"Processing text file: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if valid_chunks:
        _safe_add_texts(
            [chunk.page_content for chunk in valid_chunks],
            [chunk.metadata for chunk in valid_chunks]
        )


def ingest_file(file_path: str):
    """
    Main ingestion entry point. Routes file to appropriate processor.
    
    Args:
        file_path (str): Path to document (PDF, image, or text)
    
    Raises:
        ValueError: If file type not supported
    
    Example:
        >>> ingest_file("document.pdf")
        >>> ingest_file("screenshot.png")
    """
    logger.info(f"Ingesting: {file_path}")
    os.makedirs(settings.data_folder, exist_ok=True)

    if file_path.lower().endswith(".pdf"):
        ingest_pdf(file_path)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        ingest_image(file_path)
    elif file_path.lower().endswith(".txt"):
        ingest_txt(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        raise ValueError(f"Unsupported file type: {file_path}")

    logger.success("Ingestion complete!")
