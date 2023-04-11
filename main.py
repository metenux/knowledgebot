import os

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings

from util import create_embedding, api_key

'''
@author meten(mejg@163.com) 2023.4.11 
'''

openai.api_key = api_key


def create_context(question, max_len=1800):
    df = pd.read_csv(os.path.abspath("embeddings.csv"))

    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    q_embeddings = create_embedding(question)
    df['v'] = df.v.apply(eval).apply(np.array)
    df['distances'] = distances_from_embeddings(q_embeddings, df['v'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row['n'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["knowledge"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(model="text-davinci-003", question="请自我介绍", max_tokens=500, stop_sequence=None):
    context = create_context(question)

    try:
        response = openai.Completion.create(
            prompt=f"请根据下面知识回答问题:\n\ncontext: {context}\n\n---\n\n问题: {question}\nAnswer:",
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    print(answer_question(question="介绍一下红楼梦"))
