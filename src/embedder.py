"""
Embeds documents to a VectorDB with OpenAI API
"""
import chromadb
import os
from util import get_embedding_function

# Main function to embed local text files into ChromaDB.
def main():
    # Directory containing the local text files
    doc_dir = "C:/Users/samuel_wang/Downloads/Proxyai_mvp/ITS_Solutions.txt"
    
    # List all files in the directory
    doc_files = [f for f in os.listdir(doc_dir) if os.path.isfile(os.path.join(doc_dir, f))]
    print(f"Files to process: {doc_files}")  # List files to be processed

    # Create ChromaDB client that is persistent and specifies the path to the database
    client = chromadb.PersistentClient(path="../chromadb/test_db")
    
    # Delete the existing collection and create a new one
    client.delete_collection("local_texts_collection")
    
    # Handle existing collection
    collection_name = "local_texts_collection"
    try:
        collection = client.create_collection(name=collection_name, embedding_function=get_embedding_function())
    except chromadb.db.base.UniqueConstraintError:
        collection = client.get_collection(name=collection_name, embedding_function=get_embedding_function())
    
    # Embed each document into the ChromaDB collection
    for i, file_name in enumerate(doc_files):
        
        # Opens each file in read mode with UTF-8 encoding
        with open(os.path.join(doc_dir, file_name), 'r', encoding='utf-8') as file:
            doc_content = file.read()
            print(f"Embedding file: {file_name}")  # Show which file is being embedded

            # Adds document content, metadata, and unique ID to the collection
            collection.add(
                documents=[doc_content],
                metadatas=[{"file_name": file_name}],
                ids=[str(i)]
            )
    
    print("Documents embedded successfully.")
    print(collection.peek())  # To see the first documents in the collection

if __name__ == "__main__":
    main()