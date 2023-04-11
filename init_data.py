import pandas as pd
from pandas import DataFrame

from util import read_file, remove_newlines, get_shorted_knowledges, get_tokens2, create_embedding

'''
@author meten(mejg@163.com) 2023.4.11
'''


# 将制定目录中的文件变为知识并存储在csv文件中
def files_to_all_knowledges(files_dir_path):
    csv_tests = read_file(files_dir_path)
    df = pd.DataFrame(csv_tests, columns=['title', 'knowledge'])
    df['knowledge'] = df.title + "。 " + remove_newlines(df.knowledge)
    df.to_csv('all_knowledge.csv')


# 将对知识embedding
def create_embedding_csv():
    df = DataFrame()
    df['knowledge'] = get_shorted_knowledges()
    df['n'] = df.knowledge.apply(get_tokens2())
    df['v'] = df.knowledge.apply(lambda x: create_embedding(x))
    df.to_csv('embeddings.csv')


if __name__ == "__main__":
    print('------初始化开始------')
    files_to_all_knowledges('files')
    create_embedding_csv()
    print('------初始化结束------')
