
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings

from util import (
    get_question_embdding_with_cache,
    api_key, 
    embedding_cache
)
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances
)
'''
@author meten(mejg@163.com) 2023.4.11 
'''

openai.api_key = api_key

#创建问题上下文
def create_context(question, max_len=1800):
    q_embeddings = get_question_embdding_with_cache(question)
     # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    contexts = []
    embeddings = []
    tokens = []

    for key, value in embedding_cache.items():
        if isinstance(key, tuple):
            context = key[0]
            token = key[1]
        tokens.append(token)
        contexts.append(context)
        embeddings.append(value)

    distances = distances_from_embeddings(q_embeddings, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    returns = []
    cur_len = 0

    for i in indices_of_nearest_neighbors:
        # Add the length of the text to the current length
        cur_len += tokens[i] + 4
        print(f"distance:{distances[i]}")
        # If the context is too long or distance is too large, break
        if cur_len > max_len or distances[i] > 0.18:
            break

        # Else add it to the text that is being returned
        returns.append(contexts[i])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(model="gpt-3.5-turbo", question="请自我介绍", max_tokens=500, stop_sequence=None):

    # context = ""
    # 注掉这些内容以对比效果
    context = create_context(question)
    if(context == ''):
        print("没有找到对应知识")
        return

    try:
        response = openai.ChatCompletion.create(
            messages=[{"role": "user", "content": f"请根据下面知识回答问题:\n\ncontext: {context}\n\n---\n\n问题: {question}\nAnswer:"}],
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        # print(response["choices"][0]["text"].strip())
        print(response.choices[0].message.content)
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    answer_question(question="贾宝玉的姨母是谁")
