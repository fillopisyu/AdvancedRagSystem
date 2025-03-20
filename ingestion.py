from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


doc = [
    r"C:\Users\ismsa\OneDrive\Masaüstü\Vakıfbank Tüm BilgilerTR.docx"
]

# Then you would load the DOCX files using Docx2txtLoader
docs = [Docx2txtLoader(file).load() for file in doc]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=100
)

docs_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

retriver = vectorstore.as_retriever()
