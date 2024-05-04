#!/usr/bin/env python
# coding: utf-8

# # import、install

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('pip install transformers fugashi ipadic')
get_ipython().system('pip install unidic-lite')


# In[3]:


from typing import Dict, List, Tuple, Any
from __future__ import annotations
from functools import reduce
import operator
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import load_json_obj
from pandas_utility import read_csv_df
from pipeline import NaturalLanguageProcessing


# # クラスの定義

# In[40]:


class NotApp:
    """
    GUIがない料理検索用のクラス

    Attributes
    ----------
    _nlp : NaturalLanguageProcessing
        固有表現を抽出するオブジェクト
    _cuisine_info_dics_maker : CuisineInfoDictionariesMaker
        検索結果の料理の情報の辞書のリストを作成するオブジェクト
    _jp_label_dic : Dict[str, str]
        英語のラベルを日本語のラベルに変換する辞書
    """
    def __init__(
            self, model_name: str, cuisine_df_path: str,  unify_dics_path: str
    ):
        """
        コンストラクタ

        Parameters
        ----------
        model_name : str
            ファインチューニング済みモデル名
        cuisine_df_path : str
            料理のデータフレームが保存されているパス
        unify_dics_path : str
            表記ゆれ統一用辞書が保存されているパス
        """
        label_info_dics: Dict[str, str | List[str]] = {
            'AREA': {
                'jp': '都道府県/地方',
                'df_cols': ['Prefecture', 'Areas']
            },
            'TYPE': {
                'jp': '種類',
                'df_cols': ['Types']

            },
            'SZN': {
                'jp': '季節',
                'df_cols': ['Seasons']
            },
            'INGR': {
                'jp': '食材',
                'df_cols': ['Ingredients list']
            }
        }

        self._nlp = NaturalLanguageProcessing(model_name)
        self._cuisine_info_dics_maker = CuisineInfoDictionariesMaker(
            cuisine_df_path, unify_dics_path, label_info_dics
        )
        self._jp_label_dic = {
            en_label: dic['jp'] for en_label, dic in label_info_dics.items()
        }

    def search(self, classifying_text: str) -> pd.DataFrame | None:
        """
        料理の検索

        Parameters
        ----------
        classifying_text : str
            固有表現抽出対象

        Returns
        -------
        pd.DataFrame | None
            料理検索結果のデータフレーム
        """
        classified_words = self._nlp.classify(classifying_text)

        self._show_classified_words(classified_words)

        searched_cuisines_df = self._cuisine_info_dics_maker.create(classified_words)

        return searched_cuisines_df

    def _show_classified_words(
            self, classified_words: Dict[str, List[str]]
    ) -> None:
        """
        固有表現の表示

        Parameters
        ----------
        classified_words : Dict[str, List[str]]
            抽出結果の辞書
            キーが分類ラベル、バリューがそのラベルの文字列のリスト
        """
        classified_words = {
            self._jp_label_dic[en_label]: words
            for en_label, words in classified_words.items()
        }

        for label, words in classified_words.items():
            print(f'{label}\n　{"、".join(words)}')

        print()


class CuisineInfoDictionariesMaker:
    """
    料理検索結果の辞書のリスト作成用クラス

    Attributes
    ----------
    _cuisine_searcher : CuisineSearcher
        料理を検索するオブジェクト
    _word_unifier : WordUnifier
        抽出結果の表記ゆれを統一するオブジェクト
    """
    def __init__(
            self,
            cuisine_df_path: str,
            unify_dics_path: str,
            label_info_dics: Dict[str, str | List[str]]
    ):
        """
        コンストラクタ

        Parameters
        ----------
        cuisine_df_path : str
            料理のデータフレームが保存されているパス
        unify_dics_path : str
            表記ゆれ統一用辞書が保存されているパス
        label_info_dics : Dict[str, str  |  List[str]]
            固有表現のラベルとラベルに対する各種設定情報の辞書
        """
        self._cuisine_searcher = CuisineSearcher(
            cuisine_df_path, label_info_dics
        )
        self._word_unifier = WordUnifier(unify_dics_path)

    def create(
            self, classified_words: Dict[str, List[str]]
    ) -> pd.DataFrame | None:
        """
        料理検索結果の辞書の作成

        Parameters
        ----------
        classified_words : Dict[str, List[str]]
            ラベルと、そのラベルに分類された固有表現の辞書

        Returns
        -------
        pd.DataFrame | None
            料理検索結果のデータフレーム
        """
        unified_words = self._word_unifier.unify(classified_words)
        searched_cuisines_df = self._cuisine_searcher.search(unified_words)

        return searched_cuisines_df


class CuisineSearcher:
    """
    料理検索用のクラス

    Attributes
    ----------
    _df : pd.DataFrame
        料理のデータフレーム
    _label_to_col : Dict[str, List[str]]
        固有表現のラベルに対して、検索するデータフレームの列のリストの辞書
    _words_dic : Dict[str, List[str]]
        データフレームの列と、列に含まれる全ての要素の辞書
    """

    def __init__(
            self,
            cuisine_df_path: str,
            label_info_dics: Dict[str, str | List[str]]
    ):
        """
        コンストラクタ

        Parameters
        ----------
        cuisine_df_path : str
            料理のデータフレームが保存されているパス
        label_info_dics : Dict[str, str  |  List[str]]
            固有表現のラベルとラベルに対する各種設定情報の辞書
        """
        self._df = read_csv_df(cuisine_df_path)
        self._label_to_col = self._create_label_to_col(label_info_dics)
        self._words_dic = {
            col: self._find_words(col)
            for cols in self._label_to_col.values() for col in cols
        }

    def _create_label_to_col(
            self, label_info_dics: Dict[str, str | List[str]]
    ) -> Dict[str, List[str]]:
        """
        label_to_colの作成

        固有表現のラベルに対応したデータフレームの列を
        特定するための辞書を作成する

        Parameters
        ----------
        label_info_dics : Dict[str, str  |  List[str]]
            固有表現のラベルとラベルに対する各種設定情報の辞書

        Returns
        -------
        Dict[str, List[str]]
            固有表現のラベルに対して、検索するデータフレームの列のリストの辞書

        Raises
        ------
        ValueError
            label_info_dicsに、データフレームに存在しない列名が含まれている場合
        """
        label_to_col: Dict[str, List[str]] = {
            label: dic['df_cols'] for label, dic in label_info_dics.items()
        }

        df_cols = self._df.columns.tolist()
        for cols in label_to_col.values():
            for col in cols:
                if col not in df_cols:
                    raise ValueError(f'"{col}"という列名は存在しません')

        return label_to_col

    def _find_words(self, col: str) -> List[str]:
        """
        列に含まれる全要素の取得

        Parameters
        ----------
        col : str
            列名

        Returns
        -------
        List[str]
            列に含まれる全ての要素のリスト
        """
        words: List[str, List[str]] = self._df[col].value_counts().index.tolist()

        if isinstance(words[0], list):
            words_lst = words
            unique_words: List[str] = []

            for words in words_lst:
                for word in words:
                    if word not in unique_words:
                        unique_words.append(word)

            return unique_words

        return words

    def search(self, unified_words: Dict[str, List[str]]) -> pd.DataFrame | None:
        """
        料理の検索

        Parameters
        ----------
        unified_words : Dict[str, List[str]]
            表記ゆれが統一された固有表現の辞書

        Returns
        -------
        pd.DataFrame | None
            検索結果の料理の情報を持つデータフレーム
        """
        on_df_words_dic = self._create_on_df_words_dic(unified_words)

        if not on_df_words_dic:
            print('いずれの語彙もデータに存在しませんでした')

            return None

        searched_cuisines_df = self._create_searched_cuisines_df(on_df_words_dic)

        return searched_cuisines_df

    def _create_on_df_words_dic(
            self, unified_words: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        データフレームに存在する固有表現だけの辞書の作成

        Parameters
        ----------
        unified_words : Dict[str, List[str]]
            表記ゆれが統一された固有表現の辞書

        Returns
        -------
        Dict[str, List[str]]
            データフレームに存在する表記ゆれが統一された固有表現の辞書
        """
        on_df_words_dic = {col: [] for col in self._words_dic}
        not_on_df_words: List[str] = []

        for label, words in unified_words.items():
            search_cols = self._label_to_col[label]

            for word in words:
                not_on_df = True

                for col in search_cols:
                    if word in self._words_dic[col]:
                        on_df_words_dic[col].append(word)

                        not_on_df = False

                        break

                if not_on_df:
                    not_on_df_words.append(word)

        if not_on_df_words:
            CuisineSearcher._show_not_on_df_words(not_on_df_words)

        on_df_words_dic = {
            col: words for col, words in on_df_words_dic.items() if words
        }

        return on_df_words_dic

    @staticmethod
    def _show_not_on_df_words(not_on_df_words: List[str]) -> None:
        """
        データフレームに存在しなかった固有表現の表示

        Parameters
        ----------
        not_on_df_words : List[str]
            データフレームに存在しなかった固有表現のリスト
        """
        words = '、'.join(not_on_df_words)
        message = f'無効な語彙:　{words}'

        print(message)

    def _create_searched_cuisines_df(
            self, words_dic: Dict[str, List[str]]
    ) -> pd.DataFrame | None:
        """
        料理の情報を持つ辞書の作成

        Parameters
        ----------
        words_dic : Dict[str, List[str]]
            検索ワードのリストを持つ辞書

        Returns
        -------
        pd.DataFrame | None
            料理の情報を持つデータフレーム
        """
        condition_lst: List[pd.Series] = []

        for col, words in words_dic.items():
            condition = self._create_condition(col, words)
            condition_lst.append(condition)

        conditions = reduce(operator.and_, condition_lst)

        searched_cuisines_df = self._df.loc[conditions]

        if searched_cuisines_df.empty:
            print('検索条件が厳しすぎて、該当料理が見つかりませんでした')

            return None

        return searched_cuisines_df

    def _create_condition(self, col: str, words: List[str]) -> pd.Series:
        """
        検索条件の作成

        Parameters
        ----------
        col : str
            絞り込み対象列
        words : List[str]
            検索ワード

        Returns
        -------
        pd.Series
            該当料理の行がTrueになったboolのシリーズ
        """
        value_type = type(self._df.at[0, col])

        if value_type is list:
            condition = self._df[col].apply(
                lambda values: any(word in values for word in words)
            )

        else:
            conditions = [self._df[col] == word for word in words]
            condition = reduce(operator.or_, conditions)

        return condition


class WordUnifier:
    """
    表記ゆれ統一用のクラス

    Attributes
    ----------
    _not_unify_labels : List[str]
        表記ゆれ統一対象ではない固有表現のラベルのリスト
    _unify_dics : Dict[str, Dict[str, str]]
        ラベルと、そのラベルの固有表現の表記ゆれ統一用の辞書の辞書
    """
    _not_unify_labels = ['SZN']

    def __init__(self, unify_dics_path: str):
        """
        コンストラクタ

        Parameters
        ----------
        unify_dics_path : str
            表記ゆれ統一用辞書が保存されているパス
        """
        self._unify_dics: Dict[str, Dict[str, str]] = load_json_obj(unify_dics_path)

    def unify(
            self, classified_words: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        表記ゆれの統一

        Parameters
        ----------
        classified_words : Dict[str, List[str]]
            ラベルと、そのラベルに分類された固有表現の辞書

        Returns
        -------
        Dict[str, List[str]]
            表記ゆれが統一された固有表現の辞書
        """
        for label, words in classified_words.items():
            if label in self._not_unify_labels:
                continue

            unify_dic = self._unify_dics[label]

            unified_words = [
                unify_dic[word] if word in unify_dic else word for word in words
            ]

            classified_words[label] = unified_words

        return classified_words


# # 料理検索用オブジェクトの作成

# In[41]:


model_name = 'wolf4032/bert-japanese-token-classification-search-local-cuisine'
cuisine_df_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/02_local_cuisine_dataframe/local_cuisine_dataframe.csv'
unify_dics_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/01_unifying_dictionaries/unifying_dictionaries.json'

not_app = NotApp(model_name, cuisine_df_path, unify_dics_path)


# # 検索

# In[43]:


classifying_text = '仙豆を使った野菜料理を教えて下さい'
df = not_app.search(classifying_text)
df

