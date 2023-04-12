
import pandas as pd
import tiktoken
import pickle

from openai.embeddings_utils import (
    get_embedding,
)
'''
@author meten(mejg@163.com)2023.4.11
'''

api_key = "sk-xxx"

# input parameters
embedding_cache_path = "embedding_cache.pkl"  # embeddings will be saved/loaded here
default_embedding_engine = "text-embedding-ada-002" #"babbage-similarity"  # text-embedding-ada-002 is recommended


# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError as e:
    embeddings = []
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

def get_tokens(text: str):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

# def get_tokens2():
#     tokenizer = tiktoken.get_encoding("cl100k_base")
#     return lambda x: len(tokenizer.encode(x))

# this function will get embeddings from the cache and save them there afterward
def get_embedding_with_cache(
    text: str,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    token = get_tokens(text)
    print(f"Getting embedding for {text}")
    if (text, token) not in embedding_cache.keys():
        print("load from API...")
        # if not in cache, call API to get embedding
        embedding_cache[(text, token)] = get_embedding(text, default_embedding_engine)
        # save embeddings cache to disk after each update
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(text, token)]


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith("。") or chunk.endswith("，") or chunk.endswith("；"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def split_to_chunks(clean_text, size = 500):
    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = []
    chunks = create_chunks(clean_text, size, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    for i, chunk in enumerate(text_chunks):
        get_embedding_with_cache(chunk)
        results.append(f"{i}.{chunk}\n")
        print(results[-1])

    return results

if __name__ == "__main__":
    print()
