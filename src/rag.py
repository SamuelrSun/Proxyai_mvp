import re
import tiktoken
import chromadb
from openai import OpenAI
from util import get_embedding_function

# Retrieves the OpenAI API key used for authentication.
def get_openai_api_key():
    #return api key here

# Generates a default response using OpenAI's GPT-4o model.
def response(query, client):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ],
        temperature=0.7  # Adding temperature to control randomness
    )
    return response.choices[0].message.content

# Generates a response using OpenAI's GPT-4o model with retrieval-augmented generation (RAG).
def rag_response(query, contexts, client):
    context_string = "\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please answer the query using the context provided."},
            {"role": "user", "content": f"query: {query}. context: {context_string}"},
        ],
        temperature=0.7  # Adding temperature to control randomness
    )
    return response.choices[0].message.content

# Retrieves relevant contexts from ChromaDB based on the user's query.
def get_rag_context(query, client, max_tokens=16384):
    collection = client.get_collection(name="local_texts_collection", embedding_function=get_embedding_function())
    results = collection.query(query_texts=[query], n_results=3)
    contexts = [doc.replace("\n", "  ") for doc in results['documents'][0]]

    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    user_message = f"query: {query}"
    system_message = "You are a helpful assistant. Please answer the query using the context provided."

    user_tokens = len(tokenizer.encode(user_message))
    system_tokens = len(tokenizer.encode(system_message))

    total_tokens = user_tokens + system_tokens
    valid_contexts = []

    for context in contexts:
        context_tokens = len(tokenizer.encode(context))
        if total_tokens + context_tokens <= max_tokens:
            valid_contexts.append(context)
            total_tokens += context_tokens
        else:
            break

    return valid_contexts

# Main function to interactively handle user queries, generating default and RAG responses.
def main():
    print("Starting the RAG system...")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=get_openai_api_key())
    print("OpenAI client initialized.")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="../chromadb/test_db")
    print("ChromaDB client initialized.")
    
    while True:
        user_query = input("Ask a question (type 'exit' to quit): ").strip()
        
        if user_query.lower() == 'exit':
            print("Exiting...")
            break
        
        if not user_query:
            print("No input provided. Please ask a question.")
            continue

        # Normalize the query to handle variations
        user_query = re.sub(r'\s+', ' ', user_query.strip())
        
        contexts = get_rag_context(user_query, chroma_client)
        default_resp = response(user_query, openai_client)
        ragged_resp = rag_response(user_query, contexts, openai_client)
        
        print(f"\nQuery: {user_query}")
        print(f"Default response: {default_resp}")
        print(f"\nRAG response: {ragged_resp}")
        print()

if __name__ == "__main__":
    main()