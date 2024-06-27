from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os

embeddings = OllamaEmbeddings(model="llama3")

def youtubeSupport(youtube_url: str,store_name):
    if not youtube_url:
        return
    local_store_folder = 'localStore'
    pickle_file_path = os.path.join(local_store_folder, f"{store_name}.pkl")
    loader = YoutubeLoader.from_youtube_url(
        youtube_url, add_video_info=False
    )
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(transcript)
    print("inititting llama3 for embeads")     
    VectorStore = FAISS.from_documents(chunks, embedding=embeddings)         
    with open(pickle_file_path, "wb") as f:
        pickle.dump(VectorStore, f)
    return VectorStore
