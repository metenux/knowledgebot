import os

import openai
import pandas as pd
import tiktoken
from pandas import DataFrame

'''
@author meten(mejg@163.com)2023.4.11
'''

api_key = "sk-xxx"


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('\[\d\]', repl=' ')
    return serie


def read_file(files_dir_path: str):
    csv_texts = []
    for file in os.listdir(files_dir_path):
        with open(files_dir_path + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()
            csv_texts.append((file, text))
    return csv_texts


def get_all_knowledge() -> DataFrame:
    return pd.read_csv('all_knowledge.csv', index_col=0)


# 切分知识
def split_knowledge(text: str, max_tokens: int, splitSymbol: str):
    # Split the text into sentences
    sentences = text.split(splitSymbol)

    # Get the number of tokens for each sentence
    n_tokens = [get_tokens(" " + sentence) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append("。 ".join(chunk) + "。")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def get_tokens(text: str):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_tokens2():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return lambda x: len(tokenizer.encode(x))


def get_shorted_knowledges():
    max_tokens = 500
    knowledges = []
    df = get_all_knowledge()
    df.columns = ['title', 'knowledge']

    df['n_tokens'] = df.knowledge.apply(get_tokens2())

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['knowledge'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            knowledges += split_knowledge(row[1]['knowledge'], max_tokens, "。")

        # Otherwise, add the text to the list of shortened texts
        else:
            knowledges.append(row[1]['text'])

    return knowledges


def create_embedding(x: str):
    return openai.Embedding.create(api_key, input=x,
                                   engine='text-embedding-ada-002')['data'][0]['embedding']

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
            if chunk.endswith("。") or chunk.endswith("\n") or chunk.endswith(".") :
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
        # results.append(extract_chunk(chunk,template_prompt))
        results.append(f"{i}.{chunk}\n")
        # print(chunk)
        print(results[-1])

    return results

# def extract_chunk(document,template_prompt):
    
#     prompt=template_prompt.replace('<document>',document)

#     response = openai.Completion.create(
#     model='text-davinci-003', 
#     prompt=prompt,
#     temperature=0,
#     max_tokens=1500,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return "1." + response['choices'][0]['text']

if __name__ == "__main__":
    print()
