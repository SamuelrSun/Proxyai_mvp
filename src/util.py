"""
Utility functions for the project.
"""
import os
import json
import openai
from chromadb.utils import embedding_functions

def get_openai_api_key():
    #return os.environ.get('OPENAI_API_KEY', api key here)

"""
Initializes the embedding function which uses the given model to process text
and generate a vector capturing the semantic meaning of text
"""
def get_embedding_function():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=get_openai_api_key(),
                    model_name="text-embedding-ada-002"
                )
    return openai_ef

# Processes input text and returns high-dimensional vector representing semantic meaning of text
def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']