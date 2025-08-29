# Code for vecotr store
from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

# Function to load an already existing vector store
def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vecotrstore")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            
        else:
            logger.warning("No vector store found")

    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))

# Function to create a new vector store
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found")
        
        logger.info("Creating a new vector store")
        
        embedding_model = get_embedding_model()
        
        db = FAISS.from_documents(text_chunks, embedding_model)
        
        logger.info("Saving the new vector store")
        
        db.save_local(DB_FAISS_PATH)
        
        logger.info("New vector store saved successfully")
        
        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create new vector store", e)
        logger.error(str(error_message))
        
