import os
import faiss
import numpy as np
import v5.constants as const
from v5.logger import loggers_utils
from langchain.schema import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

logger = loggers_utils(__name__)


def process_vector_store(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_model: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    ),
    recreate_vector_store=False,
) -> BaseRetriever:
    """
    Processes documents by chunking them and creating a vector store for retrieval.

    Args:
        chunk_size (int, optional): The size of each text chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 100.
        embedding (GoogleGenerativeAIEmbeddings, optional): The embedding model used for vectorization.
            Defaults to GoogleGenerativeAIEmbeddings(model="models/embedding-001").

    Returns:
        BaseRetriever: A retriever object that can be used to fetch relevant documents based on queries.
    """
    logger.info("Starting process_vector_store function.")
    if os.path.exists("v4/backend/faiss_index") and not recreate_vector_store:
        logger.info("Loading existing FAISS index from local storage.")
        vector_store = FAISS.load_local(
            "v4/backend/faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info(
            "No existing FAISS index found or recreation requested. Processing documents."
        )
        docs = WebBaseLoader(const.lib_info_link).load()
        logger.info("Loaded documents from web source.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)
        logger.info("Split documents into chunks. Total chunks: %d", len(splits))

        embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in splits]
        )
        logger.info("Generated embeddings for documents.")

        dimension = len(embeddings[0])
        hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
        logger.info("Initialized FAISS HNSW index with dimension: %d", dimension)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        hnsw_index.add(embeddings_array)
        logger.info("Added embeddings to FAISS HNSW index.")

        vector_store = FAISS(
            embedding_function=embeddings,
            index=hnsw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        ).from_documents(splits, embedding_model)
        logger.info("Wrapped FAISS index with LangChain's FAISS vector store.")

        vector_store.save_local("v4/backend/faiss_index")
        logger.info("Saved FAISS index locally.")

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    logger.info("Converted vector store into retriever.")
    return retriever
