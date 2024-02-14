import boto3
import streamlit as st
from PIL import Image
import numpy as np
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

aws_bedrock = boto3.client(service_name="bedrock-runtime")
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=aws_bedrock)

def read_csv():
    loader = CSVLoader(file_path='messages.csv', encoding="utf-8", csv_args={
                    'delimiter': ','})
    data = loader.load()
    return data

def get_vector_store(data):
    vectorstore_faiss=FAISS.from_documents(
        data,
        embeddings
    )
    vectorstore_faiss.save_local('vector_store')

def llama2_model():
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=aws_bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

System Prompt: Use the following pieces of context to provide a 
short answer to the question. Answer within 1 or 2 statements. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. 
You are Hari Prasad Renganathan. Answer how Hari Prasad Renganathan would answer it.
<context>
{context}
</context

Question: {question}

Hari:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Linkedin Chatbot")
    
    st.header("Hari AI clone")

    avatar = Image.open('ai_clone.png')
    st.image(avatar, caption="Hari AI clone", width=100) 

    user_question = st.text_input("Ask a Question for Hari")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                data = read_csv()
                print(data)
                get_vector_store(data)
                st.success("Done")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            vector_store = FAISS.load_local("vector_store", embeddings)
            llm=llama2_model()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,vector_store,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
