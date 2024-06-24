import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="llama3")
def pdfSupport(pdf,store_name):  
    if pdf is None:
        return
    local_store_folder = 'localStore'
    pickle_file_path = os.path.join(local_store_folder, f"{store_name}.pkl")
    pdf_reader = PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # st.write(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    # generating embeaadings from llama3
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(pickle_file_path, "wb") as f:
        pickle.dump(VectorStore, f)
    return VectorStore
