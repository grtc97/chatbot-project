# Code to generate embeddings for each text chunk
from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing the HuggingFace Embedding Model")
        
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info("HuggingFace Embedding model loaded successfully")
        
        return model
    
    except Exception as e:
        error_message = CustomException("Error occured while loading the Embedding Model", e)
        logger.error(str(error_message))
        raise error_message
    
    