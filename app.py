import os
import pickle
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from youtubeHelper import youtubeSupport
from pdfHelper import pdfSupport

llm=Ollama(model="llama3")
# Sidebar content
with st.sidebar:
    st.title('CramBuddy')
    
    st.markdown('''
        ## About
        Your one-stop solution for last-minute revisions.

        - **AI-powered by Llama3**: Get precise answers and insights.
        - **YouTube Integration**: Access specific information from YouTube videos by providing links.
        - **PDF Support**: Easily extract and explore information from PDF documents.
    ''')
    add_vertical_space(10)
    option = st.selectbox('Choose your input type:', ('YouTube Video', 'PDF'))

def generateEmbeadings(input):
    if option == 'PDF':
        store_name = input.name[:-4]
    else:
        store_name = input.split('=')[-1]
    # checking in local store
    local_store_folder = 'localStore'
    pickle_file_path = os.path.join(local_store_folder, f"{store_name}.pkl")
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, "rb") as f:
            VectorStore = pickle.load(f)
        return VectorStore
    
    if option == 'PDF':
        VectorStore = pdfSupport(pdf=input,store_name=store_name)
    else:
        VectorStore = youtubeSupport(youtube_url=input,store_name=store_name)
    return VectorStore

def takeInput():
    if option == 'PDF':
        input = st.file_uploader("Upload pdf",type='pdf')  
    else:
        input = st.text_input('Enter YouTube video URL:')
    return input

def main():
    header = "Chat with your "+option
    st.header(header)
    input = takeInput()
    if input is None:
        return
    with st.spinner('Generating embeddings...'):
        VectorStore = generateEmbeadings(input)
    inputPlaceholder = "Ask questions about your "+option
    if VectorStore:
        query = st.text_input(inputPlaceholder)
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with st.spinner('Generating output ...'):
                response = chain.run(input_documents=docs, question=query) 
            st.write(response)
       
if __name__ == '__main__':
    main()
