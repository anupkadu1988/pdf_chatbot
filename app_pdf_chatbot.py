import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

load_dotenv()

## set streamlit page
st.set_page_config(page_title="Core Java Chatbot")
st.title("Learn Core Java using Chatbot")

## set sidebar configuration
sidebar=st.sidebar
sidebar.title(body="Configurations selection")

## select embedding model
embedding_model=sidebar.selectbox("Select embedding model from Huggingface:", ("select model", "all-MiniLM-L6-v2"))

## select llm model using groq
llm_model_name=sidebar.selectbox("Select LLM model from Groq", ("select model", "gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192", "llama-3.1-8b-instant"))

## pdf file picker
uploaded_pdf=sidebar.file_uploader("Select pdf file for data", type="pdf", accept_multiple_files=False)

## build llm model
def build_llm(model_name):
    groq_api_key=os.getenv("GROQ_API_KEY")
    return ChatGroq(model=model_name, groq_api_key=groq_api_key)

## read pdf documents
def get_pdf_documents(file_name):
    return PyPDFLoader(file_path=file_name).load()

## create splitters
def split_documents(documents):
    splitters=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitters.split_documents(documents=documents)

## build embeddings model
def build_embeddings_model(model_name):
    os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
    return HuggingFaceEmbeddings(model_name=model_name)

## store documents in FAISS vector store
def store_vector(split_docs, embeddings):
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

if sidebar.button("Get Chatbot ready for Q&A"):
    if embedding_model in "select model":
        st.error(f"Embedding model is not selected ... please select correct model")
    if llm_model_name in "select model":
        st.error(f"LLM model is not selected ... please select correct model")
    if uploaded_pdf is None:
        st.error(f"Knowledge base pdf is not selected ")
    else:
        with st.spinner("Setting up chatbot..."):
            pdf_documents=get_pdf_documents(uploaded_pdf.name)
            split_docs=split_documents(pdf_documents)
            embeddings=build_embeddings_model(embedding_model)
            vector_store=store_vector(split_docs, embeddings)

            ## create retriever and llm model
            st.session_state.retriever=vector_store.as_retriever()
            st.session_state.llm_model=build_llm(llm_model_name)

            # Prompt template for chat
            st.session_state.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}"
            )
        )

            st.session_state.qa_chain = (
            RunnableLambda(lambda x: {
                "context": "\n\n".join(
                    [doc.page_content for doc in st.session_state.retriever.get_relevant_documents(x["question"])]
                ),
                "question": x["question"]
            })
            | st.session_state.prompt
            | st.session_state.llm_model
        )

            # Initialize session state for chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

# Input box
if "qa_chain" in st.session_state:
    user_input = st.chat_input("Ask a question about your PDF...")
    
    if user_input:
        # Display user message and add to history
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Get response
        response = st.session_state.qa_chain.invoke({"question": user_input})

        # Display and store response
        st.chat_message("assistant").markdown(response.content)
        st.session_state.chat_history.append(AIMessage(content=response.content))

    # Display all chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
else:
    st.info("Please configure and click 'Get Chatbot ready for Q&A' from the sidebar.")
