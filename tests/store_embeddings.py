# store_embeddings.py

from langchain.embeddings.openai import OpenAIEmbeddings  # Correct import
from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from LLM_config import llm_config  # Import the LLMTestConfig class

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=llm_config.embedding_model_name)

# Initialize vector store and specify the directory to persist embeddings
vector_store = Chroma(embedding_function=embeddings, persist_directory="../embeddings")

# Load and chunk contents of the blog
loader = RecursiveUrlLoader(llm_config.data_url_or_file_path)
docs = loader.load()

# Step 2: Chunk the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add documents to the vector store
vector_store.add_documents(documents=all_splits)

print("Embeddings have been stored successfully.")
