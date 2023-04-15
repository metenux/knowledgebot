import pandas as pd
import pickle
from pandas import DataFrame

from util import split_to_chunks, embedding_cache

'''
@author meten(mejg@163.com) 2023.4.11
'''

if __name__ == "__main__":
    print('------初始化开始------')
    # file = "files/人工智能介绍"
    file = "files/红楼梦介绍"
    with open(file, "r", encoding="UTF-8") as f:
        text = f.read()
    clean_text = text.replace("  ", " ").replace("\n", " ")
    # print(clean_text)
    split_to_chunks(clean_text, 300)
    print('------初始化结束------')
