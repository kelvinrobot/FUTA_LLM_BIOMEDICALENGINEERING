# config.py
import os
import sys
import logging
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


if IN_COLAB:
    VECTORSTORE_DIR = Path("/content/drive/MyDrive/bioinformatics_tutor_ai/vectorstore")
else:
    VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"

# Confidence threshold for TutorAgent
CONFIDENCE_THRESHOLD = 0.65




if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/home/user/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
