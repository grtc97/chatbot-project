import os

HF_TOKEN = os.environ.get("HF_TOKEN")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

HUGGINGFACE_REPO_ID = "mistralai/Misrtal-7B-Insrtuct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
