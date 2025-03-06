from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import constants as const
from prompt import create_chat_prompt

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

docs = WebBaseLoader(const.lib_info_link).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
)
retriever = vectorstore.as_retriever()
chat_prompt_template = create_chat_prompt()


def generate_response(query):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
