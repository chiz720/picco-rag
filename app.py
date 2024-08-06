

import os
import my_api
import streamlit as st
import time
from langchain_community.llms import Cohere
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma

HF_TOKEN = my_api.get_hf_key()
os.environ['COHERE_API_KEY'] = my_api.get_cohere_key()
os.environ['HUGGINGFACE_API_TOKEN']=HF_TOKEN


# Initialize embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# Initialize vector store
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)

print("Triggering the retriever")
retriever = vectordb.as_retriever()

print("Adding prompt....")
# Define prompt template
template = """
User: You are AI assistant that follows instructions extremely well.
Please be truthful and give direct answers. Say "I don't know" if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Query: {question}

Assistant:
"""
llm = HuggingFaceHub(
    repo_id = "chikZ/local_rag",
    model_kwargs = {
        "max_new_tokens":512,
        "repition_penalty":1.1,
        "temperature":0.5,
        "top_p":0.9,
        "return_full_text":False
    }
)

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = ({
    "contex": retriever.with_config(run_name="Docs"),
    "question": RunnablePassthrough()
}
| prompt
| llm
| output_parser)

st.title("Chat with PDF")
choice = st.sidebar.selectbox("Choose",['Upload','Existing file'])

if choice == 'Upload':
    file_path = st.file_uploader("Upload your PDF")
    # call ingest.py and pass the path
else: 
    query = st.text_input("Enter your prompt:")
    text_container = st.empty()
    full_text = ""
    for chunk in chain.stream(query):
        words = chunk.split()
        for word in words:
            full_text+=word+ " "
            text_container.markdown(full_text)
            time.sleep(0.05)

