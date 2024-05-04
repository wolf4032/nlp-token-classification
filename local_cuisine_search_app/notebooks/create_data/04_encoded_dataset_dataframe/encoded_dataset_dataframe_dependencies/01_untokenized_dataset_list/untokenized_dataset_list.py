#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List, Dict
import random
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import load_json_obj, RandomPicker, search_strs, dump_obj_as_json


# # 関数とクラスの定義

# In[3]:


def create_and_save(
        templates_path: str, unify_dics_path: str, file_name: str, save_dir: str
) -> List[Dict[str, str | List[Dict[str, str | List[int]]]]]:
    """
    データセットの作成と保存

    Parameters
    ----------
    templates_path : str
        データセットのテンプレートが保存されているパス
    unify_dics_path : str
        表記ゆれ統一用辞書が保存されているパス
    file_name : str
        保存するデータセットのファイル名
    save_dir : str
        データセットの保存先ディレクトリ

    Returns
    -------
    List[Dict[str, str | List[Dict[str, str | List[int]]]]]
        データセット
    """
    templates = load_json_obj(templates_path)
    data_maker = DataMaker(unify_dics_path)

    random.seed(42)

    dataset = [data_maker.create(template) for template in templates]

    dump_obj_as_json(dataset, file_name, save_dir)

    return dataset


class DataMaker:
    """
    データ作成用のクラス

    Attributes
    ----------
    _pron_label : str
        代名詞のラベル
        固有表現として言語モデルに抽出させる表現ではないため、エンティティの作成対象ではない
    _word_pickers : Dict[str, RandomPicker]
        ラベルと、ラベルに応じた語彙のRandomPickerの辞書
    _tokens : List[str]
        全トークンのラベルのリスト
    """
    _pron_label = 'PRON'

    def __init__(self, unify_dics_path: str):
        """
        コンストラクタ

        _word_pickerと_tokensの作成

        Parameters
        ----------
        unify_dics_path : str
            表記ゆれ統一用辞書が保存されているパス
        """
        self._word_pickers = DataMaker._create_word_pickers(unify_dics_path)
        self._tokens = list(self._word_pickers.keys())

    @staticmethod
    def _create_word_pickers(unify_dics_path: str) -> Dict[str, RandomPicker]:
        """
        word_pickersの作成

        Parameters
        ----------
        unify_dics_path : str
            表記ゆれ統一用辞書が保存されているパス

        Returns
        -------
        Dict[str, RandomPicker]
            ラベルと、ラベルに応じた語彙のRandomPickerの辞書
        """
        unify_dics: Dict[str, Dict[str, str]] = load_json_obj(unify_dics_path)
        word_pickers: Dict[str, RandomPicker] = {}

        for label, unify_dic in unify_dics.items():
            words = [word for words in unify_dic.items() for word in words]
            word_pickers[label] = RandomPicker(words)

        word_pickers['SZN'] = RandomPicker(['春', '夏', '秋', '冬', '通年'])
        word_pickers[DataMaker._pron_label] = RandomPicker(
            ['料理', 'お料理', '郷土料理', 'レシピ', 'もの', 'やつ']
        )

        return word_pickers

    def create(
            self, template: str
    ) -> Dict[str, str | List[Dict[str, str | List[int]]]]:
        """
        データの作成

        Parameters
        ----------
        template : str
            データのテンプレート

        Returns
        -------
        Dict[str, str | List[Dict[str, str | List[int]]]]
            データ
            文章と、その文章に含まれる固有表現の情報を持つ
        """
        include_tokens = search_strs(template, self._tokens)

        text = template
        entities: List[Dict[str, str | List[int]]] = []
        for token in include_tokens:
            replace_word = self._pick_word(token)

            if token != DataMaker._pron_label:
                entity_dic = self._create_entity_dic(text, token, replace_word)
                entities.append(entity_dic)

            text = text.replace(f'[{token}]', replace_word, 1)

        data = {'text': text, 'entities': entities}

        return data

    def _pick_word(self, token: str) -> str:
        """
        具体的な語彙の取得

        特殊トークンに入れる具体的な語彙をランダムに取得する

        Parameters
        ----------
        token : str
            置き換え対象の特殊トークンのラベル

        Returns
        -------
        str
            具体的な語彙
        """
        picker = self._word_pickers[token]
        word = picker.pick()

        return word

    def _create_entity_dic(
            self, text: str, token: str, replace_word: str
    ) -> Dict[str, str | List[int]]:
        """
        entity_dicの作成

        データに含まれる特定の固有表現の情報を持つ辞書を作成する

        Parameters
        ----------
        text : str
            データの文章
        token : str
            固有表現のラベル
        replace_word : str
            固有表現

        Returns
        -------
        Dict[str, str | List[int]]
            データに含まれる特定の固有表現の情報を持つ辞書
        """
        name_start_idx = text.find('[')
        name_end_idx = name_start_idx + len(replace_word)
        span = [name_start_idx, name_end_idx]

        entity_dic = {'name': replace_word, 'span': span, 'type': token}

        return entity_dic


# # 実行

# In[4]:


templates_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list.json'
unify_dics_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/01_unifying_dictionaries/unifying_dictionaries.json'
file_name = 'untokenized_dataset_list'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list'

dataset = create_and_save(templates_path, unify_dics_path, file_name, save_dir)


# # 出力結果の確認

# In[5]:


print(f'data数: {len(dataset)}\n')

dataset

