from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama 
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceHub
import os 
import streamlit as st

__import__('pysqlite3')
import sys 
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')

HF_TOKEN = st.secrets['HUGGINGFACE_ACCESS_TnHF_TOKENKEN']
os.environ['HUGGINGFACE_API_TOKEN']=HF_TOKEN


# Initialize embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# Initialize vector store
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)

print("Triggering the retriever")
# Initialize retriever
retriever = vectordb.as_retriever()

print("Adding prompt....")
# Define prompt template
template = """
User: You are AI assistant that follows instructions extremely well.
Please be truthful and give direct answers. Say "I don't know" if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Query: {question}

Remember only return AI answer
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

st.title("Picco Rag")
