import sys
sys.modules["tensorflow"] = None
sys.modules["keras"] = None
sys.modules["tf_keras"] = None

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
if not hasattr(st.session_state, "store"):
    st.session_state.store = None

from dotenv import load_dotenv
load_dotenv()

from langchain_classic.chains.retrieval import create_retrieval_chain #for defining retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever #for creating history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #for defining document_chain
from langchain_chroma import Chroma #Chroma Vectorstore DB

#-------------------------Chat Message History relevant packages-------------------
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
#----------------------------------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
hf_embeddings = HuggingFaceEmbeddings(
                    model_name = "all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )


# Set Streamlit
st.title("Conversation RAG with PDF uploads and chat history")
st.write("Upload PDF")

# Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initializing LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Create session_id (UI)
session_id = st.text_input("Session ID", value = "default_session")

# Statefully managing the Chat History

if st.session_state.store is None:
    st.session_state.store = {}

# Upload Button (UI)
uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files=True)

# Process uploaded PDF files
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        pdf_loader = PyPDFLoader(temp_pdf)
        docs = pdf_loader.load()
        documents.extend(docs) #append documents
    
    # Splitting and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =5000, chunk_overlap = 500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)
    retriever = vectorstore.as_retriever()

    #System Prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed otherwise return it as it is"
    )

    # Prompt template
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    # Answer-Question Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know"
        "Give a descriptive answer"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    #Document chain
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

    #Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


    # Getting specific session_id's chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()

        return st.session_state.store[session_id]


    # Final chain: Conversation RAG Chain
    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_rag_chain.invoke(
            {"input": user_input},
            config = {
                "configurable": {"session_id": session_id}
            }, # this config constructs a key eg. "abc123" in `st.session.store`
        )
        st.write(st.session_state.store)
        st.write("Assistant: ", response['answer'])
        st.write("Chat History: ", session_history.messages)

#end if (if uploaded_files)

