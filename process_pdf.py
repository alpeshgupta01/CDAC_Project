#This function embeds the PDF document into a vector Database using API model
#Creates a local vector DB with ChromaDB 
#Returns vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def process_pdf(disease,doc_path, vector_dir, chunk_size, chunk_overlap, embed_model):
    # 1. Check if document exists
    if not doc_path.exists():
        raise FileNotFoundError(f"Cannot find the file at: {doc_path}")
    # 2. Read Document
    reader = PyPDFLoader(str(doc_path))
    docs = reader.load()
    # 3. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    
    # 4. Vectorization
    embeddings = HuggingFaceEmbeddings(model=embed_model)
    
    # 5.Store Embeddings 
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name=disease,
        persist_directory=vector_dir.as_posix() 
    )
    return vectorstore
