from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from LLM_config import llm_config  # Import the LLMTestConfig class

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=llm_config.embedding_model_name)

# Initialize vector store
vector_store = Chroma(embedding_function=embeddings)

# Load and chunk contents of the blog
loader = RecursiveUrlLoader(llm_config.data_url_or_file_path)
docs = loader.load()

# Step 2: Chunk the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)