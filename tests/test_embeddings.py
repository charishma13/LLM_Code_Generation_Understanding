# query_embeddings.py

from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Load embeddings and vector store from disk
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(persist_directory="../embeddings", embedding_function=embeddings)

# Initialize OpenAI Chat model
llm_openAI = ChatOpenAI(model="o3-mini",
                        reasoning_effort="high")

# Define a function to query the embeddings and get an answer
def query_embeddings(query: str):
    """Query the stored embeddings to get a general answer."""
    # Perform a similarity search in the vector store
    retrieved_docs = vector_store.similarity_search(query, k=2)

    # Combine the retrieved documents' contents for prompt
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message_content = f"""
            "You are an expert in molecular simulations in chemistry and energy calculations for molecules. "
            "Use the retrieved context (if available) to answer the question.\n\n"
            {docs_content}
        """

    # Prepare the prompt for the model
    prompt = [SystemMessage(system_message_content)]
    
    # Get a response from the model
    response = llm_openAI.invoke(prompt)

    return response


# Test the function with a sample query
if __name__ == "__main__":
    query = "How advancements in areas like artificial intelligence or bioscience are influencing simulation methods and energy calculations?"
    response = query_embeddings(query)
    print(f"Response: {response}")
