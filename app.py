from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama 
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os 
import streamlit as st

#HF_TOKEN = st.secrets['HUGGINGFACE_ACCESS_TnHF_TOKENKEN']
#os.environ['HUGGINGFACE_API_TOKEN']=HF_TOKEN
HF_TOKEN = "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"

# Initialize embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# Initialize vector store
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)

print("Triggering the retriever")
# Initialize retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

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
prompt = ChatPromptTemplate.from_template(template)

print("Running Model Ollama...")
# Initialize LLM
llm = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler]))

# Define chain
chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm 
    | StrOutputParser()
)

# Invoke chain
print("AI Response")
query = "What parameters can be measured using PiCCO? State the physiological reference ranges"
print(chain.invoke(query))