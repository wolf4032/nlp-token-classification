#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List, Dict
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from pandas_utility import save_csv_df


# # クラスの定義

# In[3]:


class DataframeMaker:
    """
    データフレームを作成するクラス

    Attributes
    ----------
    _expr_dic_lst : List[Dict[str, str | List[str]]]
        各トークンに付随する表現と、それに類する表現のリストを持つ辞書のリスト
    """
    _expr_dic_lst: List[Dict[str, str | List[str]]] = [
        {
            'token': '[AREA]',
            'expr': 'で食べられる',
            'similars': ['で食べられている']
        },
        {
            'token': '[SZN]',
            'expr': 'に食べられる',
            'similars': ['に食べられている']
        },
        {
            'token': '[INGR]',
            'expr': 'を使った',
            'similars': ['を使用した', 'が使われている']
        }
    ]

    @staticmethod
    def create_and_save(
            read_path: str, remove_label: str, file_name: str, save_dir: str
    ) -> pd.DataFrame:
        """
        データフレームの作成と保存

        Parameters
        ----------
        read_path : str
            似た表現を追加するデータフレームが保存されているパス
        remove_label : str
            データセットの作成に使わないクラスのラベル
        file_name : str
            保存するデータフレームのファイル名
        save_dir : str
            データフレームの保存先ディレクトリ

        Returns
        -------
        pd.DataFrame
            似た表現を追加されたデータフレーム
        """
        original_df = DataframeMaker._create_original_df(read_path, remove_label)

        similar_exprs_dic = DataframeMaker._create_similar_exprs_dic()

        df = DataframeMaker._create_df(original_df, similar_exprs_dic)

        save_csv_df(df, file_name, save_dir)

        return df

    @staticmethod
    def _create_original_df(read_path: str, remove_label: str) -> pd.DataFrame:
        """
        似た表現が追加されていないデータフレームの作成

        データセットの作成に使わないクラスの行の削除も行う

        Parameters
        ----------
        read_path : str
            似た表現を追加するデータフレームが保存されているパス
        remove_label : str
            データセットの作成に使わないクラスのラベル

        Returns
        -------
        pd.DataFrame
            似た表現が追加されていないデータフレーム
        """
        df = pd.read_csv(read_path)

        label_col = df.columns[1]

        df = df[df[label_col] != remove_label]
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _create_similar_exprs_dic() -> Dict[str, List[str]]:
        """
        似た表現の辞書の作成

        ある表現とそれに類する表現の辞書を作成する

        Returns
        -------
        Dict[str, List[str]]
            似た表現が追加されていないデータフレームで使われている表現の中で、
            似た表現を持つものと、それに似た表現のリストの辞書
        """
        similar_exprs_dic = {}

        for expr_dic in DataframeMaker._expr_dic_lst:
            token: str = expr_dic['token']

            original_expr: str = token + expr_dic['expr']
            similar_exprs = [
                token + similar for similar in expr_dic['similars']
            ]

            similar_exprs_dic[original_expr] = similar_exprs

        return similar_exprs_dic

    @staticmethod
    def _create_df(
            original_df: pd.DataFrame,
            similar_exprs_dic: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        似た表現が追加されたデータフレームの作成

        Parameters
        ----------
        original_df : pd.DataFrame
            似た表現が追加されていないデータフレーム
        similar_exprs_dic : Dict[str, List[str]]
            似た表現の辞書

        Returns
        -------
        pd.DataFrame
            似た表現が追加されたデータフレーム
        """
        expr_col, class_col = original_df.columns
        df_dic = {expr_col: [], class_col: []}

        def add_similar_exprs(
                row: pd.Series,
                similar_exprs_dic=similar_exprs_dic,
                df_dic=df_dic,
                expr_col=expr_col,
                class_col=class_col
        ) -> None:
            """
            .apply()用に、row以外のデフォルト値を指定するためのメソッド
            """
            DataframeMaker._add_similar_exprs(
                row, similar_exprs_dic, df_dic, expr_col, class_col
            )

        original_df.apply(add_similar_exprs, axis=1)

        df = pd.DataFrame(df_dic)

        return df

    @staticmethod
    def _add_similar_exprs(
            row: pd.Series,
            similar_exprs_dic: Dict[str, List[str]],
            df_dic: Dict[str, List[str]],
            expr_col: str,
            class_col: str
    ) -> None:
        """
        似た表現の追加

        渡された文頭表現に類する全ての表現とそのクラスをdf_dicに追加する

        Parameters
        ----------
        row : pd.Series
            似た表現の作成対象の行
        similar_exprs_dic : Dict[str, List[str]]
            似た表現の辞書
        df_dic : Dict[str, List[str]]
            似た表現を追加されたデータフレームの元となる辞書
            文頭表現と、各文頭表現のクラスのラベルを持つ
        expr_col : str
            文頭表現列の列名
        class_col : str
            クラス列の列名
        """
        original_expr = row[0]
        label = row[1]

        exprs: List[str] = [original_expr]
        for expr in similar_exprs_dic.keys():
            if expr in original_expr:
                similar_exprs = similar_exprs_dic[expr]
                DataframeMaker._update_exprs(expr, similar_exprs, exprs)

        labels = [label for _ in range(len(exprs))]

        df_dic[expr_col].extend(exprs)
        df_dic[class_col].extend(labels)

    @staticmethod
    def _update_exprs(
            expr: str, similar_exprs: List[str], exprs: List[str]
    ) -> None:
        """
        文頭表現のリストの更新

        似た表現が追加されていないデータフレームの特定の行の文頭表現に
        似たすべての表現をもつリストに表現を追加する

        Parameters
        ----------
        expr : str
            似た表現に置き換えられる表現
        similar_exprs : List[str]
            exprの置き換え対象のリスト
        exprs : List[str]
            文頭表現のリスト
        """
        fmr_exprs = exprs.copy()
        exprs.clear()

        for fmr_expr in fmr_exprs:
            exprs.append(fmr_expr)

            other_exprs = [
                fmr_expr.replace(expr, similar_expr)
                for similar_expr in similar_exprs
            ]

            exprs.extend(other_exprs)


# # 実行

# In[4]:


read_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe/start_of_sentences_dataframe_v2_classified.csv'
remove_label = 'remove'
file_name = 'start_of_sentences_dataframe_v3_added_similar_expressions'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe'

df = DataframeMaker.create_and_save(read_path, remove_label, file_name, save_dir)


# # 出力結果の確認

# In[5]:


df

