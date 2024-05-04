#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List
import random
from __future__ import annotations
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import search_strs
from pandas_utility import save_csv_df


# # 関数とクラスの定義

# In[3]:


def create_and_save(
        read_path: str, file_name: str, save_dir: str
) -> pd.DataFrame:
    """
    データフレームの作成と保存

    Parameters
    ----------
    read_path : str
        データフレームが保存されているパス
    file_name : str
        保存するデータフレームのファイル名
    save_dir : str
        データフレームの保存先ディレクトリ

    Returns
    -------
    pd.DataFrame
        カンマと並列表現が追加されたデータフレーム
    """
    df = pd.read_csv(read_path)

    expr_col = df.columns[0]
    random.seed(42)
    df[expr_col] = df[expr_col].apply(DataframeEditor.edit)

    save_csv_df(df, file_name, save_dir)

    return df


class DataframeEditor:
    """
    データフレームにカンマと並列表現を追加するクラス

    Attributes
    ----------
    _special_tokens : List[str]
        固有表現の特殊トークン
    _not_para_token : str
        固有表現じゃない特殊トークン
        並列化の対象にならない
    _add_comma_tokens : List[str]
        カンマ追加対象の特殊トークン
    """
    _special_tokens = ['[AREA]', '[TYPE]', '[SZN]', '[INGR]']
    _not_para_token = '[PRON]'

    _add_comma_tokens = _special_tokens + [_not_para_token]

    @staticmethod
    def edit(expr: str) -> str:
        """
        文頭表現へのカンマと並列表現の追加

        Parameters
        ----------
        expr : str
            文頭表現

        Returns
        -------
        str
            カンマと並列表現が追加された文頭表現
        """
        include_tokens = search_strs(expr, DataframeEditor._add_comma_tokens)

        if len(include_tokens) > 1:
            expr = DataframeEditor._add_comma(expr, include_tokens[1:])  # ※１

        if DataframeEditor._not_para_token in include_tokens:
            include_tokens.remove(DataframeEditor._not_para_token)

        expr = Parallelizer.parallelize(expr, include_tokens)

        return expr

    @staticmethod
    def _add_comma(expr: str, include_tokens: List[str]) -> str:
        """
        カンマの追加

        Parameters
        ----------
        expr : str
            文頭表現
        include_tokens : List[str]
            文頭表現に含まれている特殊トークン
            先頭のトークンはこのリストに含まない

        Returns
        -------
        str
            カンマが追加された文頭表現
        """
        for token in include_tokens:
            random_num = random.random()

            if random_num < 0.2:
                comma_added_expr = expr.replace(token, f'、{token}')

                return comma_added_expr  # ※２

        return expr


class Parallelizer:
    """
    並列表現を追加するクラス

    Attributes
    ----------
    _parallel_words : List(str)
        並列表現に使用する接続詞のリスト
    """
    _parallel_words = ['と', 'か', 'または', 'あるいは']

    @staticmethod
    def parallelize(expr: str, include_tokens: List[str]) -> str:
        """
        並列表現の追加

        Parameters
        ----------
        expr : str
            文頭表現
        include_tokens : List[str]
            文頭表現に含まれている特殊トークン
            固有表現じゃない特殊トークンはこのリストに含まない

        Returns
        -------
        str
            並列表現が追加された文頭表現
        """
        for token in include_tokens:
            random_num = random.random()

            if random_num < 0.25:
                para_word = random.choice(Parallelizer._parallel_words)
                para_word = ParallelWordMaker.add_comma(para_word)

                expr = expr.replace(token, f'{token}{para_word}{token}')

                return expr  # ※３

        return expr


class ParallelWordMaker:
    """
    接続詞にカンマを追加するクラス

    Attributes
    ----------
    _comma_positions_dic : Dict[str, int]
        カンマの位置と、その位置にする確率の辞書
    _comma_positions : List[str]
        カンマの位置のリスト
    _positions_weights : List[int]
        各カンマの位置を採用する確率のリスト
    """
    _comma_positions_dic = {
        'front': 10, 'back': 20, 'double': 15, 'none': 55
    }

    _comma_positions = list(_comma_positions_dic.keys())
    _positions_weights = list(_comma_positions_dic.values())

    @staticmethod
    def add_comma(para_word: str) -> str:
        """
        カンマの追加

        Parameters
        ----------
        para_word : str
            接続詞

        Returns
        -------
        str
            カンマが追加された接続詞
        """
        comma_position_lst = random.choices(
            ParallelWordMaker._comma_positions,
            ParallelWordMaker._positions_weights,
            k=1
        )
        comma_position = comma_position_lst[0]

        if comma_position == 'front':
            para_word = '、' + para_word

        elif comma_position == 'back':
            para_word = para_word + '、'

        elif comma_position == 'double':
            para_word = '、' + para_word + '、'

        return para_word


# # 実行

# In[4]:


read_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe/start_of_sentences_dataframe_v3_added_similar_expressions.csv'
file_name = 'start_of_sentences_dataframe_v4_added_comma_and_parallelize'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe'

df = create_and_save(read_path, file_name, save_dir)


# # 出力結果の確認

# In[5]:


df


# # メモ

# ※１
# - カンマ追加処理は、ランダムに選択したトークンの”直前に”追加するようにしている
# - 先頭のトークンに対してカンマ追加処理がされると、文頭がカンマになってしまうため、`include_tokens[1:]`とすることで、先頭のトークンをカンマ追加対象から除外した

# ※２
# - 今回のデータセットの文章の長さから、各文章に追加するカンマは一つまでが自然な表現に近そうだと思った

# ※３
# - 文脈から並列表現であることを判別させるには、いくつかの文章に、一つの並列表現があれば十分ではないかと考えた
