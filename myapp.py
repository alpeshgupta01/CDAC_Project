#This is a custom chatbot application to demo RAG architecture using 
#Custom Docs + HuggingFace Model.
import streamlit as st
import os
from get_strict_rag_chain import get_strict_rag_chain
from get_open_rag_chain import get_open_rag_chain
from pathlib import Path

# --- Page setup ---
st.set_page_config(page_title="Preventive HealthCare Chat Demo", page_icon="💬")

st.title("💬 CDAC AI BATCH PROJECT DEMO")
st.write("A simple Streamlit app to demo RAG. Answers queries related to common occuring diseases in India"
" Anaemia,Asthma,Covid-19,Dengue,Diabetes,HyperTension,Malaria,Tuberculosis and Typhoid.")
user_choice = st.toggle("Use Open knowledge", value=False) # Default to False (No)

current_dir = Path.cwd() #Get Current Working Directory
vector_dir = current_dir / os.getenv("VECTOR_DIR") # Read the Vector DB Base directory path 
knn=int(os.getenv("KNN")) #Neareset Neighbours for similarity search
gpt_model_creativity=int(os.getenv("HF_GPT_MODEL_CREATIVITY")) #Chat Model to creativity
gpt_model_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS")) #New tokens for gpt model


@st.cache_resource(show_spinner=False)  # Add the caching decorator
def load_strict_rag():
    return get_strict_rag_chain(
        knn,
        st.secrets["HF_EMBEDDING_MODEL"],
        st.secrets["HF_GPT_MODEL"],
        vector_dir,
        gpt_model_creativity,
        gpt_model_new_tokens
    )

@st.cache_resource(show_spinner=False)  # Add the caching decorator
def load_open_rag():
    return get_open_rag_chain(
        knn,
        st.secrets["HF_EMBEDDING_MODEL"],
        st.secrets["HF_GPT_MODEL"],
        vector_dir,
        gpt_model_creativity,
        gpt_model_new_tokens
    )
# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display chat history ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if user_choice:
        rag_response = load_open_rag().invoke({"input": prompt})
    else:
        rag_response = load_strict_rag().invoke({"input": prompt})

    # Add assistant response
    st.session_state["messages"].append({"role": "assistant", "content": rag_response['answer']})
    st.chat_message("assistant").write(rag_response['answer'])