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


from typing import List, Dict, Tuple
import math
import pandas as pd
from transformers import BertJapaneseTokenizer
from transformers.tokenization_utils_base import BatchEncoding

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import load_json_obj
from pandas_utility import save_csv_df


# # クラスの定義

# In[4]:


class DatasetMaker:
    """
    データセット作成用のクラス
    """
    @staticmethod
    def create_and_save(
            untokenized_dataset_path: str,
            model_name: str,
            labels_dic_path: str,
            file_name: str,
            save_dir: str
    ) -> pd.DataFrame:
        """
        データセットの作成と保存

        Parameters
        ----------
        untokenized_dataset_path : str
            トークン化されていないデータセットが保存されているパス
        model_name : str
            事前学習済み言語モデルの名前
            トークナイザーの設定に使う
        labels_dic_path : str
            特殊トークンのラベルとそのidの辞書が保存されているパス
        file_name : str
            保存するデータセットのファイル名
        save_dir : str
            データセットの保存先ディレクトリ

        Returns
        -------
        pd.DataFrame
            エンコード済みのデータセット
        """
        untokenized_dataset = load_json_obj(untokenized_dataset_path)

        texts = [data['text'] for data in untokenized_dataset]
        tokens_max_len = DatasetMaker._decide_tokens_max_len(texts)

        data_maker = DataMaker(model_name, tokens_max_len, labels_dic_path)

        dataset: List[BatchEncoding] = []
        for untokenized_data in untokenized_dataset:
            data = data_maker.create(untokenized_data)

            if data:
                dataset.append(data)

        data_maker.show_unk_words_and_remove_texts()

        dataset = pd.DataFrame(
            data=dataset, columns = ['input_ids', 'attention_mask', 'labels']
        )

        save_csv_df(dataset, file_name, save_dir)

        return dataset

    @staticmethod
    def _decide_tokens_max_len(texts: List[str]) -> int:  # ※１
        """
        tokens_max_lenの決定

        各データのトークン数の決定

        Parameters
        ----------
        texts : List[str]
            トークン化されていないデータセットの入力文のリスト

        Returns
        -------
        int
            最大トークン数
        """
        max_len_of_text = 0

        for text in texts:
            len_of_text = len(text)

            if len_of_text > max_len_of_text:
                max_len_of_text = len_of_text

        log_of_max_len = math.log2(max_len_of_text)
        rounded_up_log = math.ceil(log_of_max_len)

        tokens_max_len = 2 ** rounded_up_log

        return tokens_max_len


class DataMaker:
    """
    データ作成用のクラス

    Attributes
    ----------
    _sep_token : str
        一文の終わりを示す特殊トークン
    _unk_token : str
        トークナイザーが知らない語彙用の特殊トークン
    _tokenizer: BertJapaneseTokenizer
        トークナイザー
    _tokens_max_len : int
        最大トークン数
    _unk_words: List[str]
        トークナイザーが知らなかった語彙のリスト
    _labels_maker : LabelsMaker
        正解ラベルのリスト作成用のオブジェクト
    _remove_texts: List[str]
        データセットに使わない文章のリスト
    """
    _sep_token = '[SEP]'
    _unk_token = '[UNK]'

    def __init__(
            self, model_name: str, tokens_max_len: int, labels_dic_path: str
    ):
        """
        コンストラクタ

        Parameters
        ----------
        model_name : str
            事前学習済み言語モデルの名前
            トークナイザーの設定に使う
        tokens_max_len : int
            最大トークン数
        labels_dic_path : str
            特殊トークンのラベルとそのidの辞書が保存されているパス
        """
        self._tokenizer: BertJapaneseTokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self._tokens_max_len = tokens_max_len
        self._unk_words: List[str] = []
        self._labels_maker = LabelsMaker(labels_dic_path)
        self._remove_texts: List[str] = []

    def create(
            self,
            untokenized_data: Dict[str, str | List[Dict[str, str | List[int]]]]
    ) -> BatchEncoding | None:
        """
        データの作成

        Parameters
        ----------
        untokenized_data : Dict[str, str  |  List[Dict[str, str  |  List[int]]]]
            トークン化されていない学習データ
            入力文と、抽出対象固有表現の情報を持つ辞書

        Returns
        -------
        BatchEncoding | None
            エンコード済みの学習データ
            トークン化の区切り位置が良くなかった場合はNone
        """
        text: str = untokenized_data['text']

        unlabeled_data = self._tokenizer.encode_plus(
            text,
            max_length=self._tokens_max_len,
            padding='max_length',
            return_token_type_ids=False
        )

        input_ids: List[int] = unlabeled_data['input_ids']
        tokens = self._decode(input_ids, text)

        entity_infos: List[Dict[str, str | List[int]]] = untokenized_data['entities']
        labels = self._labels_maker.create(
            tokens, entity_infos, self._tokens_max_len, self._remove_texts
        )

        if labels:
            unlabeled_data.update({'labels': labels})

            data = unlabeled_data

            return data

        else:
            return None

    def _decode(self, input_ids: List[int], text: str) -> List[str]:
        """
        デコード

        input_idsをトークンのリストに変換する

        Parameters
        ----------
        input_ids : List[int]
            入力文の各トークンのidのリスト
        text : str
            トークン化されていない入力文

        Returns
        -------
        List[str]
            トークンのリスト
        """
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        tokens = self._remove_extra_tokens_and_strs(tokens)

        if self._unk_token in tokens:
            tokens = Unknown.restore(
                tokens, text, self._unk_token, self._unk_words
            )

        return tokens

    def _remove_extra_tokens_and_strs(self, tokens: List[str]) -> List[str]:
        """
        余分なトークンと文字列の削除

        トークン化されていない文章の文字数に、tokensの文字数をそろえる

        Parameters
        ----------
        tokens : List[str]
            トークンのリスト

        Returns
        -------
        List[str]
            余分なトークンと文字列が削除されたトークンのリスト
        """
        sep_token_idx = tokens.index(self._sep_token)
        tokens = tokens[1:sep_token_idx]  # ※２
        tokens = [token.replace('##', '') for token in tokens]  # ※３

        return tokens

    def show_unk_words_and_remove_texts(self) -> None:
        """
        トークナイザーが知らなかった語彙とデータセットに採用しない入力文の表示
        """
        print('\nトークナイザーが知らない語彙')
        unk_words_str = '、'.join(self._unk_words)
        print(f'　{unk_words_str}')

        print('\n削除した文章')
        for remove_text in self._remove_texts:
            print(f'　{remove_text}')

        print(f'\n削除した文章数: {len(self._remove_texts)}')


class Unknown:
    """
    [UNK]トークンに関する処理を担うヘルパークラス

    正解ラベルのリストを作成するために、全てのトークンの元の文字数の情報が必要

    Attributes
    ----------
    _sep : str
        分割用文字列
    """
    _sep = '[sep]'

    @staticmethod
    def restore(
            tokens: List[str], text: str, unk_token: str, unk_words: List[str]
    ) -> List[str]:
        """
        トークンのリストの復元

        Parameters
        ----------
        tokens : List[str]
            [UNK]トークンを含むトークンのリスト
        text : str
            tokensのトークン化前の文字列
        unk_token : str
            [UNK]トークン
        unk_words : List[str]
            トークナイザーが知らなかった語彙のリスト

        Returns
        -------
        List[str]
            [UNK]トークンが復元されたトークンのリスト
        """
        include_unk_words = Unknown._restore_unk_words(tokens, text, unk_token)

        for unk_word in include_unk_words:
            unk_token_idx = tokens.index(unk_token)
            tokens[unk_token_idx] = unk_word

            if unk_word not in unk_words:
                unk_words.append(unk_word)

        return tokens

    @staticmethod
    def _restore_unk_words(
            tokens: List[str], text: str, unk_token: str
    ) -> List[str]:
        """
        [UNK]トークンの復元

        Parameters
        ----------
        tokens : List[str]
            [UNK]トークンを含むトークンのリスト
        text : str
            tokensのトークン化前の文字列
        unk_token : str
            [UNK]トークン

        Returns
        -------
        List[str]
            textに含まれていて、トークナイザーが知らなかった語彙のリスト
        """
        decoded_text = ''.join(tokens)

        strs_without_unk = Unknown._str_to_lst(decoded_text, unk_token)

        unk_words_str = text.replace('？', '?')  # ※４
        for string in strs_without_unk:
            unk_words_str = unk_words_str.replace(string, Unknown._sep)

        unk_words = Unknown._str_to_lst(unk_words_str, Unknown._sep)

        return unk_words

    @staticmethod
    def _str_to_lst(string: str, split_str: str) -> List[str]:
        """
        文字列をリストへ変換

        Parameters
        ----------
        string : str
            文字列
        split_str : str
            区切り文字

        Returns
        -------
        List[str]
            リスト
        """
        string = string.strip(split_str)  # ※５
        lst = string.split(split_str)

        return lst


class LabelsMaker:
    """
    正解ラベルのリスト作成用のクラス

    Attributes
    ----------
    _label2id_dic : Dict[str, int]
        ラベルをidに変換する辞書
    _other_token_id
        抽出対象じゃないトークンのラベルのid
    """
    def __init__(self, labels_dic_path: str):
        """
        コンストラクタ

        Parameters
        ----------
        labels_dic_path : str
            特殊トークンのラベルとそのidの辞書が保存されているパス
        """
        id2label_dic = load_json_obj(labels_dic_path)
        self._label2id_dic: Dict[str, int] = {
            label: id for id, label in id2label_dic.items()
        }
        self._other_token_id = list(id2label_dic.keys())[0]

    def create(
            self,
            tokens: List[str],
            entity_infos: List[Dict[str, str | List[int]]],
            tokens_max_len: int,
            remove_texts: List[str]
    ) -> List[int] | None:
        """
        ラベルのリストの作成

        トークンの区切り位置が良くなかった場合は作成しない

        Parameters
        ----------
        tokens : List[str]
            トークンのリスト
        entity_infos : List[Dict[str, str  |  List[int]]]
            tokensに含まれる固有表現の情報の辞書のリスト
        tokens_max_len : int
            最大トークン数
        remove_texts : List[str]
            データセットに使わない文章のリスト

        Returns
        -------
        List[int] | None
            正解ラベルのidのリスト
            トークン化の区切り位置が良くなかった場合はNone
        """
        token_start_idxs, token_end_idxs = Index.create_start_end_idxs(tokens)
        ts_idxs = token_start_idxs
        te_idxs = token_end_idxs

        entity_spans: List[List[int]] = [
            entity_info['span'] for entity_info in entity_infos
        ]
        entity_start_idxs, entity_end_idxs = Index.create_start_end_idxs(entity_spans)
        es_idxs = entity_start_idxs
        ee_idxs = entity_end_idxs

        if Index.is_idxs_match(ts_idxs, te_idxs, es_idxs, ee_idxs):
            entity_types: List[str] = [
                entity_info['type'] for entity_info in entity_infos
            ]
            labels = self._create_labels(
                ts_idxs, te_idxs, es_idxs, ee_idxs, entity_types, tokens_max_len
            )

            return labels

        else:
            remove_text = '　'.join(tokens)
            remove_texts.append(remove_text)

            return None

    def _create_labels(
            self,
            ts_idxs: List[int],
            te_idxs: List[int],
            es_idxs: List[int],
            ee_idxs: List[int],
            entity_types: List[str],
            tokens_max_len: int
    ) -> List[int]:
        """
        ラベルのリストの作成

        Parameters
        ----------
        ts_idxs : List[int]
            入力文に対する、全トークンの開始位置のインデックスのリスト
        te_idxs : List[int]
            入力文に対する、全トークンの終了位置のインデックスのリスト
        es_idxs : List[int]
            入力文に対する、全固有表現の開始位置のインデックスのリスト
        ee_idxs : List[int]
            入力文に対する、全固有表現の終了位置のインデックスのリスト
        entity_types: List[str]
            入力文に含まれる全固有表現の種類のリスト
        tokens_max_len : int
            最大トークン数

        Returns
        -------
        List[int]
            ラベルのidのリスト
        """
        labels = [self._other_token_id] * tokens_max_len

        for es_idx, ee_idx, entity_type in zip(es_idxs, ee_idxs, entity_types):
            entity_begin_token_idx = ts_idxs.index(es_idx) + 1
            entity_last_token_idx = te_idxs.index(ee_idx) + 1

            begin_token_label_id = self._label2id_dic[f'B-{entity_type}']

            labels[entity_begin_token_idx] = begin_token_label_id

            if entity_begin_token_idx != entity_last_token_idx:
                inside_token_label_id = self._label2id_dic[f'I-{entity_type}']

                inside_token_idxs = slice(
                    entity_begin_token_idx + 1, entity_last_token_idx + 1
                )
                id_num = entity_last_token_idx - entity_begin_token_idx

                labels[inside_token_idxs] = [inside_token_label_id] * id_num

        return labels


class Index:
    @staticmethod
    def create_start_end_idxs(
            tokens_or_entity_spans: List[str] | List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        開始位置と終了位置のインデックスのリストの作成

        Parameters
        ----------
        tokens_or_entity_spans : List[str] | List[List[int]]
            トークンのリストか、全固有表現の開始位置と終了位置のリスト

        Returns
        -------
        Tuple[List[int], List[int]]
            開始位置のインデックスのリストと、
            終了位置のインデックスのリストのタプル
        """
        if isinstance(tokens_or_entity_spans[0], str):
            return Index._create_token_idxs(tokens_or_entity_spans)

        else:
            return Index._create_entity_idxs(tokens_or_entity_spans)

    @staticmethod
    def _create_token_idxs(tokens: List[str]) -> Tuple[List[int], List[int]]:
        """
        全トークンの開始位置と終了位置のインデックスのリストの作成

        Parameters
        ----------
        tokens : List[str]
            トークンのリスト

        Returns
        -------
        Tuple[List[int], List[int]]
            トークンの開始位置のインデックスのリストと、
            終了位置のインデックスのリストのタプル
        """
        start_idxs = []
        end_idxs = []

        current_idx = 0
        for token in tokens:
            start_idx = current_idx
            end_idx = current_idx + len(token)

            start_idxs.append(start_idx)
            end_idxs.append(end_idx)

            current_idx = end_idx

        return start_idxs, end_idxs

    @staticmethod
    def _create_entity_idxs(
            entity_spans: List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        全固有表現の開始位置と終了位置のインデックスのリストの作成

        Parameters
        ----------
        entity_spans : List[List[int]]
            全固有表現の開始位置と終了位置のインデックスのリスト

        Returns
        -------
        Tuple[List[int], List[int]]
            全固有表現の開始位置のインデックスのリストと、
            終了位置のインデックスのリストのタプル
        """
        start_idxs = []
        end_idxs = []

        for entity_span in entity_spans:
            entity_start_idx = entity_span[0]
            entity_end_idx = entity_span[1]

            start_idxs.append(entity_start_idx)
            end_idxs.append(entity_end_idx)

        return start_idxs, end_idxs

    @staticmethod
    def is_idxs_match(
            ts_idxs: List[int],
            te_idxs: List[int],
            es_idxs: List[int],
            ee_idxs: List[int]
    ) -> bool:
        """
        トークンと固有表現の開始位置と終了位置の確認

        位置がそろっていれば、正解ラベルを付けることができる

        Parameters
        ----------
        ts_idxs : List[int]
            入力文に対する、全トークンの開始位置のインデックスのリスト
        te_idxs : List[int]
            入力文に対する、全トークンの終了位置のインデックスのリスト
        es_idxs : List[int]
            入力文に対する、全固有表現の開始位置のインデックスのリスト
        ee_idxs : List[int]
            入力文に対する、全固有表現の終了位置のインデックスのリスト

        Returns
        -------
        bool
            そろっていればTrue、そろっていなければFalse
        """
        token_idxs_lst = [ts_idxs, te_idxs]
        entity_idxs_lst = [es_idxs, ee_idxs]

        for token_idxs, entity_idxs in zip(token_idxs_lst, entity_idxs_lst):
            if any(entity_idx not in token_idxs for entity_idx in entity_idxs):
                return False

        return True


# # 実行

# In[5]:


untokenized_dataset_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list.json'
model_name = 'cl-tohoku/bert-base-japanese-v2'
labels_dic_path = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/03_labels_dictionary/labels_dictionary.json'
file_name = 'encoded_dataset_dataframe'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe'

dataset = DatasetMaker.create_and_save(
    untokenized_dataset_path, model_name, labels_dic_path, file_name, save_dir
)


# # 出力結果の確認

# In[6]:


dataset


# # メモ

# ※１
# - 深層学習のフレームワークは、2のべき乗のシーケンス長に最適化されていることが多いようなので、このような処理でデータの`max_length`を決めることにした

# ※２
# - 抽出対象トークンに正解ラベルを付与する処理のために、`[CLS]`、`[SEP]`、`[PAD]`を`tokens`から省く

# ※３
# - サブワードに付く`##`も、省いておかないと、正解ラベルを付与するための処理で各トークンの`span`と、抽出対象の語彙の`span`にずれが生じてしまう

# ※４
# - 今回の処理に使ったトークナイザーは、語彙に半角の`?`は持っているが、全角の`？`は持っていない
# - `encode_plus(...)`に渡されたtext内の全角の`？`は、`convert_ids_to_tokens(input_ids)`によって半角の`?`として出力される
# - `unk_words_str.replace(string, Unknown._sep)`で置き換えられるように、textの全角の`？`は半角の`?`に変えておく必要がある

# ※５
# - stringの先頭が`unk_token`だと、`string.split(split_str)`の最初の要素が空文字（`''`）になってしまう
# - stringの末尾が`Unknown._sep`だと、`string.split(split_str)`の最後の要素も空文字（`''`）になってしまう
