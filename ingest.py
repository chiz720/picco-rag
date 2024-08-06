from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma

from my_api import get_hf_key

HF_TOKEN = get_hf_key()

def persist_dir(file_path):
    data = PyPDFLoader(file_path)
    print("Loading data....")
    content = data.load()
    print("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)
    chunks = splitter.split_documents(content)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="mixedbread-ai/mxbai-embed-large-v1"
    )
    
    print("Save to db...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./db"
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    data = "/Users/idrischikophe/Desktop/picco-rag/data /picco_technology.pdf"
    persist_dir(data)
