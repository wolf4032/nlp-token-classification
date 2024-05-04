#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List, Dict

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import dump_obj_as_json


# # 関数の定義

# In[3]:


def create_and_save(
        labels: List[str], file_name: str, save_path: str
) -> Dict[int, str]:
    """
    BERTモデルの特殊トークンとそのIDの辞書の作成

    Parameters
    ----------
    labels : List[str]
        抽出対象のラベル
    file_name : str
        辞書のファイル名
    save_path : str
        辞書の保存先ディレクトリ

    Returns
    -------
    Dict[int, str]
        特殊トークンとそのIDの辞書
    """
    dic = {0: 'O'}

    id_num = 1
    for label in labels:
        dic[id_num] = 'B-' + label

        id_num += 1

        dic[id_num] = 'I-' + label

        id_num += 1

    dump_obj_as_json(dic, file_name, save_path)

    return dic


# # 実行

# In[5]:


labels = ['AREA', 'TYPE', 'SZN', 'INGR']
file_name = 'labels_dictionary'
save_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/03_labels_dictionary'

dic = create_and_save(labels, file_name, save_path)
dic

