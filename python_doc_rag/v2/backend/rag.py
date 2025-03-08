from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


import constants as const
from prompt import create_chat_prompt

load_dotenv()


def get_chat_model() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a ChatGoogleGenerativeAI model instance.

    The model used is 'gemini-2.0-flash' with a temperature of 0 for deterministic responses.
    It allows an unlimited number of tokens per response (`max_tokens=None`),
    has no request timeout (`timeout=None`), and retries twice in case of failure.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini 2.0 Flash chat model.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm


def process_vector_store(
    chunk_size: int = 500,
    chunk_overlap: int = 200,
    embedding: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    ),
) -> BaseRetriever:
    """
    Processes documents by chunking them and creating a vector store for retrieval.

    Args:
        chunk_size (int, optional): The size of each text chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 200.
        embedding (GoogleGenerativeAIEmbeddings, optional): The embedding model used for vectorization.
            Defaults to GoogleGenerativeAIEmbeddings(model="models/embedding-001").

    Returns:
        BaseRetriever: A retriever object that can be used to fetch relevant documents based on queries.
    """
    # Load documents from a web source
    docs = WebBaseLoader(const.lib_info_link).load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    # Create a vector store with the embedded chunks
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
    )

    # Convert the vector store into a retriever
    retriever = vectorstore.as_retriever()

    return retriever


llm = get_chat_model()
retriever = process_vector_store()
chat_prompt_template = create_chat_prompt()


def generate_response(
    query: str,
    retriever: BaseRetriever,
    llm: ChatGoogleGenerativeAI,
    chat_prompt_template: ChatPromptTemplate,
) -> str:
    """
    Generates a response to a given query using a retrieval-augmented chain.

    Args:
        query (str): The user's input query.
        retriever (BaseRetriever): A retriever instance for fetching relevant context.
        llm (ChatGoogleGenerativeAI): The language model for generating responses.
        chat_prompt_template (ChatPromptTemplate): A prompt template to structure input for the model.

    Returns:
        str: The generated response as a string.
    """
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
