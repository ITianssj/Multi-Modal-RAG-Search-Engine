"""
Vision Processing Module

Handles image description generation using Groq's vision capabilities.
Converts images to base64 and sends to LLM for semantic understanding.
"""

from groq import Groq
from config import settings
from loguru import logger
import base64
import mimetypes


client = Groq(api_key=settings.groq_api_key)


def describe_image(image_path: str) -> str:
    """
    Generate a semantic description of an image using Groq's vision model.
    
    Args:
        image_path (str): Path to the image file (PNG, JPG, JPEG)
    
    Returns:
        str: Detailed description of the image content, or empty string on error
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: Logs any API errors and returns empty string
    
    Example:
        >>> desc = describe_image("screenshot.png")
        >>> print(desc)
        "This screenshot shows a dashboard with..."
    """
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

        # Call Groq vision API
        response = client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Include all text, charts, diagrams, handwritten notes, and layout."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=512
        )
        
        description = response.choices[0].message.content
        return str(description).strip() if description else ""
    except Exception as e:
        logger.error(f"Failed to describe image {image_path}: {e}")
        return ""