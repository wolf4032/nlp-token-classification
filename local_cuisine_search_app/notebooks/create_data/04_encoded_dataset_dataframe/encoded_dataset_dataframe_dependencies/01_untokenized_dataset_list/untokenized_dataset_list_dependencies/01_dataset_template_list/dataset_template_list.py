#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List, Dict, Tuple
import random
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import dump_obj_as_json, RandomPicker


# # 関数とクラスの定義

# In[3]:


def create_and_save(
        sos_df_path: str, eos_df_path: str, file_name: str, save_dir: str
) -> List[str]:
    """
    データセットのテンプレートの作成と保存

    Parameters
    ----------
    sos_df_path : str
        文頭表現のデータフレームが保存されているパス
    eos_df_path : str
        文末表現のデータフレームが保存されているパス
    file_name : str
        保存するデータセットテンプレートのファイル名
    save_dir : str
        データセットテンプレートの保存先ディレクトリ

    Returns
    -------
    List[str]
        データセットテンプレート
    """
    sos_df = pd.read_csv(sos_df_path)
    templates_maker = TemplatesMaker(eos_df_path, sos_df)

    templates = []
    random.seed(42)
    sos_df.apply(
        templates_maker.extend_templates,
        args=(templates,),
        axis=1
    )

    dump_obj_as_json(templates, file_name, save_dir)

    return templates


class TemplatesMaker:
    """
    テンプレート作成用クラス

    Attributes
    ----------
    _almighty_class : str
        全てのクラスの文末表現に繋がるクラス
    _class_col : str
        文末表現データフレームのクラス列の列名
    _eos_col : str
        文末表現データフレームの文末表現列の列名
    _eos_df : pd.DataFrame
        文末表現データフレーム
    _pickers : Dict[str, RandomPicker]
        文末表現のクラスと、クラスに応じた文末表現のRandomPickerの辞書
    """
    _almighty_class = 'almighty'
    _class_col = 'class'
    _eos_col = '文末表現'

    def __init__(self, eos_df_path: str, sos_df: pd.DataFrame):
        """
        コンストラクタ

        _eos_df、_pickersの作成と、文頭表現、文末表現のクラスのチェックを行う

        Parameters
        ----------
        eos_df_path : str
            文末表現データフレームが保存されているパス
        sos_df : pd.DataFrame
            文頭表現データフレーム
        """
        self._eos_df = TemplatesMaker._create_eos_df(eos_df_path)
        eos_classes = TemplatesMaker._create_classes(self._eos_df)

        TemplatesMaker._check_sos_and_eos_classes(sos_df, eos_classes)

        self._pickers = self._create_pickers(eos_classes)

    @staticmethod
    def _create_eos_df(eos_df_path: str) -> pd.DataFrame:
        """
        文末表現データフレームの作成

        Parameters
        ----------
        eos_df_path : str
            文末表現データフレームが保存されているパス

        Returns
        -------
        pd.DataFrame
            文末表現データフレーム
        """
        eos_df = pd.read_csv(eos_df_path)
        eos_df.fillna('', inplace=True)  # ※１

        return eos_df

    @staticmethod
    def _create_classes(df: pd.DataFrame) -> List[str]:
        """
        全クラスのリストの作成

        渡されたデータフレームに含まれる全てのクラスを持ったリストを作る

        Parameters
        ----------
        df : pd.DataFrame
            文頭表現か、文末表現のデータフレーム

        Returns
        -------
        List[str]
            データフレームに含まれる全クラスのリスト
        """
        class_value_counts = df[TemplatesMaker._class_col].value_counts()
        classes: List[str] = class_value_counts.keys().tolist()

        return classes

    @staticmethod
    def _check_sos_and_eos_classes(
            sos_df: pd.DataFrame, eos_classes: List[str]
    ) -> None:
        """
        クラスの確認

        文頭表現のクラスに、文末表現にないクラスが含まれていないか確認する
        各文頭表現は、後に続くことができる文末表現のクラスを指定している

        Parameters
        ----------
        sos_df : pd.DataFrame
            文頭表現データフレーム
        eos_classes : List[str]
            文末表現の全クラス

        Raises
        ------
        ValueError
            ある文頭表現が、文末表現にないクラスを指定していた場合
        """
        sos_classes = TemplatesMaker._create_classes(sos_df)
        sos_classes.remove(TemplatesMaker._almighty_class)

        extra_sos_classes = [
            sos_cls for sos_cls in sos_classes if sos_cls not in eos_classes
        ]

        if extra_sos_classes:
            extra_sos_classes_str = '、'.join(extra_sos_classes)

            raise ValueError(
                f'{extra_sos_classes_str}というクラスは文末表現にありません'
            )

    def _create_pickers(self, eos_classes: List[str]) -> Dict[str, RandomPicker]:
        """
        _pickersの作成

        Parameters
        ----------
        eos_classes : List[str]
            文末表現の全クラス

        Returns
        -------
        Dict[str, RandomPicker]
            文末表現のクラスと、クラスに応じた文末表現のRandomPickerの辞書
        """
        pickers: Dict[str, RandomPicker] = {}

        for eos_class in eos_classes:
            class_idxs = self._create_class_idxs(eos_class)
            pickers[eos_class] = RandomPicker(class_idxs)

        return pickers

    def _create_class_idxs(self, eos_class: str) -> List[int]:
        """
        class_idxsの作成

        文末表現の各クラスに該当する文末表現データフレームの
        行インデックスのリストを作成する

        Parameters
        ----------
        eos_class : str
            文末表現のクラス

        Returns
        -------
        List[int]
            eos_classに該当する行インデックスのリスト
        """
        is_class_row = self._eos_df[self._class_col] == eos_class
        class_idxs: List[int] = self._eos_df[is_class_row].index.tolist()

        return class_idxs

    def extend_templates(self, row: pd.Series, templates: List[str]) -> None:
        """
        templatesの拡張

        Parameters
        ----------
        row : pd.Series
            文頭表現とその文頭表現に続く文末表現のクラスの情報を持つ行
        templates : List[str]
            テンプレートのリスト
        """
        sos, sos_class = row.values

        if sos_class == self._almighty_class:
            for eos_class in self._pickers.keys():
                self._append_template(sos, eos_class, templates)

        else:
            self._append_template(sos, sos_class, templates)

    def _append_template(
            self, sos: str, s_class: str, templates: List[str]
    ) -> None:
        """
        テンプレートの作成と追加

        Parameters
        ----------
        sos : str
            文頭表現
        s_class : str
            表現のクラス
        templates : List[str]
            テンプレートのリスト
        """
        eos = self._pick_eos(s_class)

        template = sos + eos
        templates.append(template)

    def _pick_eos(self, s_class: str) -> str:
        """
        文末表現の参照

        Parameters
        ----------
        s_class : str
            表現のクラス

        Returns
        -------
        str
            ランダムに参照した文末表現
        """
        picker = self._pickers[s_class]
        eos_idx: int = picker.pick()
        eos: str = self._eos_df.at[eos_idx, self._eos_col]

        return eos


# # 実行

# In[4]:


sos_df_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe/start_of_sentences_dataframe_v4_added_comma_and_parallelize.csv'
eos_df_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/02_end_of_sentences_dataframe/end_of_sentences_dataframe.csv'
file_name = 'dataset_template_list'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list'

template_list = create_and_save(sos_df_path, eos_df_path, file_name, save_dir)


# # 出力結果の確認

# In[5]:


print(f'template数: {len(template_list)}\n')
template_list


# # メモ

# ※１
# - 文頭表現だけで一つの文章として成立する可能性がある場合、文末表現のデータフレームに、`文末表現`列が空の行を用意している
# - 空の行が読み込まれると、その行の`文末表現`列の値は`NaN`となってしまう
# - `NaN`のままだと`template = sos + eos`でエラーが起きるため、`NaN`を空文字（`''`）に置き換えている
