#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import Dict, List, Iterator, Tuple
import itertools
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from pandas_utility import save_csv_df


# # 関数とクラスの定義

# In[3]:


def create_and_save(file_name: str, save_dir: str) -> pd.DataFrame:
    """
    データフレームの作成と保存

    文頭表現のデータフレームを作成して保存する

    Parameters
    ----------
    file_name : str
        保存するファイル名
    save_dir : str
        保存先のディレクトリ

    Returns
    -------
    pd.DataFrame
        文頭表現のデータフレーム
    """
    df = DataframeMaker.create()

    save_csv_df(df, file_name, save_dir)

    return df


class DataframeMaker:
    """
    文頭表現のデータフレームを作成するクラス

    Attributes
    ----------
    _token_particles_dic : Dict[str, Tuple[str]]
        各トークンと、それに続く表現の辞書
    _pronoun_token : str
        代名詞のトークン
    _not_add_pronoun_tokens : Tuple[str]
        直後に代名詞が続かないトークン
    _not_last_particles : Tuple[str]
        最後のトークンには、これのいずれかの文字列を含む表現を使わない
    _not_double_strs : Tuple[str]
        これらの文字列は、一つの文頭表現で２回以上は採用しない
    """
    _token_particles_dic: Dict[str, Tuple[str]] = {
        '[AREA]': ('の', 'で', 'で食べられる', 'に'),
        '[TYPE]': ('で', 'に'),
        '[SZN]': ('の', 'に', 'に食べられる'),
        '[INGR]': ('を使った',)
    }
    _pronoun_token: str = '[PRON]'
    _not_add_pronoun_tokens: Tuple[str] = ('[TYPE]',)  # ※１
    _not_last_particles: Tuple[str] = ('に', 'で')
    _not_double_strs: Tuple[str] = ('に', '食')

    @classmethod
    def create(cls) -> pd.DataFrame:
        """
        データフレームの作成

        作成した全トークンの組み合わせのリストをもとに、
        文頭表現のデータフレームを作成する

        Returns
        -------
        pd.DataFrame
            文頭表現のデータフレーム
        """
        tokens = list(cls._token_particles_dic.keys())

        token_combs = DataframeMaker._create_token_combs(tokens)

        all_start_of_sentences = DataframeMaker._create_all_start_of_sentences(
            token_combs
        )

        df = pd.DataFrame(all_start_of_sentences)

        return df

    @staticmethod
    def _create_token_combs(tokens: List[str]) -> List[Tuple[str]]:
        """
        トークンの組み合わせの作成

        トークンの全ての組み合わせのイテレータを作成する

        Parameters
        ----------
        tokens : List[str]
            組み合わせ作成対象のトークン

        Returns
        -------
        List[Tuple[str]]
            トークンの全組み合わせのリスト
        """
        token_combs: List[Tuple[str]] = []

        max_tokens_num = len(tokens) + 1

        for tokens_num in range(1, max_tokens_num):
            combs: Iterator[Tuple[str]] = itertools.combinations(
                tokens, tokens_num
            )

            for comb in combs:
                permutations: Iterator[Tuple[str]] = itertools.permutations(comb)

                token_combs.extend(permutations)

        return token_combs

    @staticmethod
    def _create_all_start_of_sentences(
            token_combs: List[Tuple[str]]
    ) -> List[str]:
        """
        全ての文頭表現の作成

        全ての文頭表現のリストを作成する

        Parameters
        ----------
        token_combs : List[Tuple[str]]
            トークンの全組み合わせ

        Returns
        -------
        List[str]
            全ての文頭表現のリスト
        """
        all_start_of_sentences: List[str] = []

        for comb in token_combs:
            start_of_sentences = DataframeMaker._create_start_of_sentences(comb)

            all_start_of_sentences.extend(start_of_sentences)

        return all_start_of_sentences

    @classmethod
    def _create_start_of_sentences(cls, comb: Tuple[str]) -> List[str]:
        """
        文頭表現の作成

        渡されたトークンの組み合わせに該当する全ての文頭表現を作成する

        Parameters
        ----------
        comb : Tuple[str]
            トークンの組み合わせ

        Returns
        -------
        List[str]
            文頭表現のリスト
        """
        start_of_sentences: List[str] = []
        soss = start_of_sentences

        last_token_idx = len(comb) - 1

        for token_idx, token in enumerate(comb):
            if token_idx == last_token_idx:
                soss = DataframeMaker._add_last_strs(soss, token)

            else:
                particles = cls._token_particles_dic[token]
                tokens_added_particle = [token + partcl for partcl in particles]

                if token_idx == 0:
                    soss = tokens_added_particle

                else:
                    soss = DataframeMaker._add_following_strs(
                        soss, tokens_added_particle
                    )

        return soss

    @classmethod
    def _add_last_strs(cls, soss: List[str], token: str) -> List[str]:
        """
        最後の表現の追加

        特定の組み合わせの最後のトークンの表現を作成し、追加する

        Parameters
        ----------
        soss : List[str]
            作成途中の文頭表現のリスト
        token : str
            最後に追加するトークン

        Returns
        -------
        List[str]
            文頭表現のリスト
        """
        if token in cls._not_add_pronoun_tokens:
            if soss:
                soss = [sos + token for sos in soss]

            else:
                soss.append(token)

        else:
            particles = cls._token_particles_dic[token]
            tokens_added_particle = [
                token + partcl for partcl in particles
                if partcl not in cls._not_last_particles
            ]

            if soss:
                soss = DataframeMaker._add_following_strs(
                    soss, tokens_added_particle
                )
                soss = [sos + cls._pronoun_token for sos in soss]

            else:
                soss = [
                    t_a_p + cls._pronoun_token for t_a_p in tokens_added_particle
                ]

        return soss

    @classmethod
    def _add_following_strs(
            cls, soss: List[str], following_strs: List[str]
    ) -> List[str]:
        """
        続く表現の追加

        作成途中の文頭表現に続くトークンと付随する全ての表現を追加する

        Parameters
        ----------
        soss : List[str]
            作成途中の文頭表現
        following_strs : List[str]
            トークンとそれに続く表現のリスト

        Returns
        -------
        List[str]
            following_strsが付与された文頭表現のリスト
        """
        former_soss = soss.copy()
        soss.clear()

        for fmr_sos in former_soss:
            not_double_strs = [
                string for string in cls._not_double_strs if string in fmr_sos
            ]

            if not_double_strs:
                the_following_strs = [
                    f_str for f_str in following_strs
                    if not any(n_d_str in f_str for n_d_str in not_double_strs)
                ]

            else:
                the_following_strs = following_strs

            soss.extend([fmr_sos + f_str for f_str in the_following_strs])

        return soss


# # 実行

# In[4]:


file_name = 'start_of_sentences_dataframe_v1'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/01_start_of_sentences_dataframe'

df = create_and_save(file_name, save_dir)


# # 出力結果の確認

# In[5]:


df


# # メモ

# ※１
# - `'[PRON]'`に入るのは、`'料理'`とか、`'もの'`とか、`'やつ'`とか、検索対象そのものを示す言葉
# - `'[TYPE]'`に入るのは、`'飯料理'`とか、`'肉料理'`とか、料理の種類を示す言葉
# - `'[TYPE]'`の直後に、`'[PRON]'`が続くような表現はおかしい（「飯料理の料理」とか、「肉料理のもの」とか）ので、`_not_add_pronoun_tokens`で`'[PRON]'`が続かないトークンを指定する
