#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import List, Tuple, Dict
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from pandas_utility import save_csv_df


# # クラスの定義

# In[3]:


class DataframeMaker:
    """
    データフレーム作成用クラス

    Attributes
    ----------
    _expr_col : str
        文末表現列の列名
    _class_col : str
        クラス列の列名
    """
    _expr_col = '文末表現'
    _class_col = 'class'

    @staticmethod
    def create_and_save(
            read_path: str, sheet_names: List[str], file_name: str, save_dir: str
    ) -> pd.DataFrame:
        """
        文末表現のデータフレームの作成と保存

        Parameters
        ----------
        read_path : str
            Excelファイルの保存先パス
        sheet_names : List[str]
            各表現のシート名のリスト
            ラベル同士の繋がりの関係を示すシートの名前は不要だが、
            ○○_infoという形式にしておく必要がある
            クラスの一覧のシートの名前は'classes'としておく必要がある
        file_name : str
            保存するデータフレームのファイル名
        save_dir : str
            データフレームの保存先ディレクトリ

        Returns
        -------
        pd.DataFrame
            文末表現のデータフレーム
        """
        patterns, all_labels = PatternsAndLabels.create(read_path, sheet_names)

        classes = DataframeMaker._create_classes(read_path)
        default_class = classes[0]
        not_default_classes = classes[1:]

        expr_dfs_dic = ExpressionDataframeDictionary.create(
            read_path, sheet_names, all_labels, default_class, not_default_classes
        )

        df = DataframeMaker._create(patterns, expr_dfs_dic, default_class)

        save_csv_df(df, file_name, save_dir)

        return df

    @staticmethod
    def _create_classes(read_path: str) -> List[str]:
        """
        クラスのリストの作成

        Parameters
        ----------
        read_path : str
            Excelファイルの保存先パス

        Returns
        -------
        List[str]
            クラスのリスト
        """
        classes_df = pd.read_excel(
            io=read_path, sheet_name='classes', header=None
        )

        classes = classes_df[0].values.tolist()

        return classes

    @staticmethod
    def _create(
            patterns: List[List[Tuple[str, str]]],
            expr_dfs_dic: Dict[str, pd.DataFrame],
            default_class: str
    ) -> pd.DataFrame:
        """
        文末表現のデータフレームの作成

        Parameters
        ----------
        patterns : List[List[Tuple[str, str]]]
            どのデータフレームのどのラベルが、
            どのデータフレームのどのラベルに繋がるかの関係を示したリストのリスト
            タプルの中身は(シート名, ラベル)
        expr_dfs_dic : Dict[str, pd.DataFrame]
            各表現のシート名と、そのデータフレームの辞書
            データフレームは表現の列と、クラスの列の２列で構成されている
        default_class : str
            デフォルトのクラス

        Returns
        -------
        pd.DataFrame
            文末表現のデータフレーム
        """
        dic = {DataframeMaker._expr_col: [], DataframeMaker._class_col: []}

        for pattern in patterns:
            exprs, classes = DataframeMaker._create_exprs_and_classes(
                pattern, expr_dfs_dic, default_class
            )

            dic[DataframeMaker._expr_col].extend(exprs)
            dic[DataframeMaker._class_col].extend(classes)

        df = pd.DataFrame(dic)

        return df

    @staticmethod
    def _create_exprs_and_classes(
            pattern: List[Tuple[str, str]],
            expr_dfs_dic: Dict[str, pd.DataFrame],
            default_class: str
    ) -> Tuple[List[str], List[str]]:
        """
        文末表現のリストとクラスのリスト作成

        各文末表現と同じインデックスのクラスが、その文末表現のクラス

        Parameters
        ----------
        pattern : List[Tuple[str, str]]
            どのデータフレームのどのラベルが、
            どのデータフレームのどのラベルに繋がるかの関係を示したリスト
            タプルの中身は(シート名, ラベル)
        expr_dfs_dic : Dict[str, pd.DataFrame]
            各表現のシート名と、そのデータフレームの辞書
            データフレームは表現の列と、クラスの列の２列で構成されている
        default_class : str
            デフォルトのクラス

        Returns
        -------
        Tuple[List[str], List[str]]
            文末表現のリストとクラスのリストのタプル
        """
        expr_class_dic = {}
        for next_df_name, next_label in pattern:
            next_df = expr_dfs_dic[next_df_name]
            DataframeMaker._update_expr_class_dic(
                next_df, next_label, expr_class_dic, default_class
            )

        exprs = list(expr_class_dic.keys())
        classes = list(expr_class_dic.values())

        return exprs, classes

    @staticmethod
    def _update_expr_class_dic(
            next_df: pd.DataFrame,
            next_label: str,
            expr_class_dic: Dict[str, str],
            default_class: str
    ) -> None:
        """
        表現の辞書の更新

        Parameters
        ----------
        next_df : pd.DataFrame
            次の表現のデータフレーム
        next_label : str
            次のラベル
        expr_class_dic : Dict[str, str]
            現状の各表現とそのクラスの辞書
        default_class : str
            デフォルトのクラス
        """
        next_str_class_df = next_df.loc[next_df[1] == next_label, [0, 2]]

        if expr_class_dic:
            fmr_expr_class_dic = expr_class_dic.copy()
            expr_class_dic.clear()

            for fmr_expr, fmr_class in fmr_expr_class_dic.items():
                next_str_class_df.apply(
                    DataframeMaker._update_expr_and_class,
                    args=(fmr_expr, fmr_class, expr_class_dic, default_class),
                    axis=1
                )

        else:
            next_str_class_df.apply(
                DataframeMaker._initialize_expr_class_dic,
                args=(expr_class_dic,),
                axis=1
            )

    @staticmethod
    def _initialize_expr_class_dic(
            row: pd.Series, expr_class_dic: Dict[str, str]
    ) -> None:
        """
        表現の辞書の初期化

        Parameters
        ----------
        row : pd.Series
            最初の表現とそのクラスの情報を持つ行
        expr_class_dic : Dict[str, str]
            各表現とそのクラスの辞書
            渡された時点では空
        """
        first_str = row.values[0]
        class_label = row.values[1]
        expr_class_dic[first_str] = class_label

    @staticmethod
    def _update_expr_and_class(
            row: pd.Series,
            fmr_expr: str,
            fmr_class: str,
            expr_class_dic: Dict[str, str],
            default_class: str
    ) -> None:
        """
        表現とそのクラスの更新

        Parameters
        ----------
        row : pd.Series
            続く表現とそのクラスの情報を持つ行
        fmr_expr : str
            これまでの表現
        fmr_class : str
            これまでの表現のクラス
        expr_class_dic : Dict[str, str]
            各表現とそのクラスの辞書
        default_class : str
            デフォルトのクラス

        Raises
        ------
        ValueError
            デフォルトではないクラス同士が競合した場合
        """
        next_str = row.values[0]
        next_class = row.values[1]

        new_expr = fmr_expr + next_str

        if next_class != fmr_class:
            if fmr_class == default_class:
                new_class = next_class

            elif next_class == default_class:
                new_class = fmr_class

            else:
                raise ValueError(
                    f'''デフォルトではないクラス同士が競合しています
                    {fmr_expr = }, {fmr_class = },
                    {next_str = }, {next_class = }
                    '''
                )

        else:
            new_class = fmr_class

        expr_class_dic[new_expr] = new_class


class PatternsAndLabels:
    """
    全パターンと全ラベルを作成するクラス

    Attributes
    ----------
    _label_col : str
        ラベル列の列名
    _not_followed_label : str
        表現が続かず、そこで終わるラベルであることを示す文字列
    _follow_all_labels : str
        続く表現のデータフレームの全てのラベルがその対象であることを示す文字列
    """
    _label_col = 'label'
    _not_followed_label = 'end'
    _follow_all_labels = 'all'

    @staticmethod
    def create(
            read_path: str, sheet_names: List[str]
    ) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
        """
        全パターンと全ラベルの作成

        Parameters
        ----------
        read_path : str
            Excelファイルの保存先パス
        sheet_names : List[str]
            各表現のシート名のリスト
            このリストの各要素に"_info"を付与して読み込む

        Returns
        -------
        Tuple[List[List[Tuple[str, str]]], List[str]]
            全パターンのリストと、全ラベルのリストのタプル
        """
        sheet_names = [name + '_info' for name in sheet_names]
        dfs_dic = PatternsAndLabels._create_dfs_dic(read_path, sheet_names)

        patterns = PatternsAndLabels._create_patterns(dfs_dic)
        all_labels = PatternsAndLabels._create_all_labels(dfs_dic)

        return patterns, all_labels

    @staticmethod
    def _create_dfs_dic(
            read_path: str, sheet_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        ラベルの関係を示すデータフレームの辞書の作成

        各表現データフレームの各ラベルの繋がりの関係を示す、
        データフレームの辞書を作成する

        Parameters
        ----------
        read_path : str
            Excelファイルの保存先パス
        sheet_names : List[str]
            各表現のシート名のリスト
            このリストの各要素に"_info"を付与して読み込む

        Returns
        -------
        Dict[str, pd.DataFrame]
            表現データフレーム名と、そのデータフレームのラベルに続く
            他のデータフレームと、そのラベルを示したデータフレームの辞書
        """
        read_dfs_dic: Dict[str, pd.DataFrame] = pd.read_excel(
            io=read_path, sheet_name=sheet_names
        )

        dfs_dic = {}
        for df_name, df in read_dfs_dic.items():
            df_name = df_name.replace('_info', '')
            df[PatternsAndLabels._label_col].fillna(method='ffill', inplace=True)

            dfs_dic[df_name] = df

        return dfs_dic

    @staticmethod
    def _create_patterns(
            dfs_dic: Dict[str, pd.DataFrame]
    ) -> List[List[Tuple[str, str]]]:
        """
        全パターンの作成

        Parameters
        ----------
        dfs_dic : Dict[str, pd.DataFrame]
            表現データフレーム名と、そのデータフレームのラベルに続く
            他のデータフレームと、そのラベルを示したデータフレームの辞書

        Returns
        -------
        List[List[Tuple[str, str]]]
            どのデータフレームのどのラベルが、
            どのデータフレームのどのラベルに繋がるかの関係を示したリストのリスト
        """
        all_patterns = []
        first_df_name, first_df = list(dfs_dic.items())[0]

        first_df.apply(
            PatternsAndLabels._extend_patterns,
            args=(first_df_name, dfs_dic, all_patterns),
            axis=1
        )

        return all_patterns

    @staticmethod
    def _extend_patterns(
            label_info: pd.Series,
            df_name: str,
            dfs_dic: Dict[str, pd.DataFrame],
            all_patterns: List[List[Tuple[str, str]]],
            patterns: List[Tuple[str, str]] = []
    ) -> None:
        """
        パターンの更新

        パターンへ次のデータフレームとそのラベルのタプルを追加する

        Parameters
        ----------
        label_info : pd.Series
            次のラベル、次のラベルに続くデータフレーム、
            そのデータフレームにある次のラベルに続くラベルの情報を持った行
        df_name : str
            次のデータフレーム（次のラベルに続くデータフレームではない）
        dfs_dic : Dict[str, pd.DataFrame]
            表現データフレーム名と、そのデータフレームのラベルに続く
            他のデータフレームと、そのラベルを示したデータフレームの辞書
        all_patterns : List[List[Tuple[str, str]]]
            全パターンのリスト
        patterns : List[Tuple[str, str]]
            パターンのリスト

        Returns
        -------
        None
            次のラベルに続くデータフレームがない場合、Noneを返して処理を終了する
        """
        label, follow_df_name, follow_labels_str = label_info
        patterns.append((df_name, label))

        if follow_df_name == PatternsAndLabels._not_followed_label:
            all_patterns.append(patterns.copy())
            patterns.pop()

            return None

        follow_df = dfs_dic[follow_df_name]

        if follow_labels_str == PatternsAndLabels._follow_all_labels:
            follow_label_rows = follow_df

        else:
            follow_labels = follow_labels_str.split(',')
            is_follow_labels = follow_df[PatternsAndLabels._label_col].isin(
                follow_labels
            )  # ※１
            follow_label_rows = follow_df[is_follow_labels]

        follow_label_rows.apply(
            PatternsAndLabels._extend_patterns,
            args=(follow_df_name, dfs_dic, all_patterns, patterns),
            axis=1
        )

        patterns.pop()

    @staticmethod
    def _create_all_labels(dfs_dic: Dict[str, pd.DataFrame]) -> List[str]:
        """
        全ラベルのリストの作成

        Parameters
        ----------
        dfs_dic : Dict[str, pd.DataFrame]
            表現データフレーム名と、そのデータフレームのラベルに続く
            他のデータフレームと、そのラベルを示したデータフレームの辞書

        Returns
        -------
        List[str]
            全ラベルの辞書
        """
        all_labels = []
        for df in dfs_dic.values():
            df_labels = df[PatternsAndLabels._label_col].values.tolist()

            all_labels.extend(df_labels)

        all_labels = list(dict.fromkeys(all_labels))  # ※２

        return all_labels


class ExpressionDataframeDictionary:
    """
    各表現のデータフレームの辞書作成用クラス
    """
    @staticmethod
    def create(
            read_path: str,
            sheet_names: List[str],
            all_labels: List[str],
            default_class: str,
            not_default_classes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        各表現のデータフレームの辞書の作成

        Parameters
        ----------
        read_path : str
            Excelファイルの保存先パス
        sheet_names : List[str]
            各表現のシート名のリスト
        all_labels : List[str]
            全ラベルのリスト
        default_class : str
            デフォルトのクラス
            Excelの'classes'シートの最初の行
        not_default_classes : List[str]
            デフォルトではないクラスのリスト
            Excelの'classes'シートの最初以外の全ての行

        Returns
        -------
        Dict[str, pd.DataFrame]
            各表現のシート名と、そのデータフレームの辞書
            データフレームは表現の列と、クラスの列の２列で構成されている
        """
        df_dic: Dict[str, pd.DataFrame] = pd.read_excel(
            io=read_path, sheet_name=sheet_names, header=None
        )
        default_label = all_labels[0]

        for df_name, df in df_dic.items():
            df = ExpressionDataframeDictionary._fixup(
                df_name, df, all_labels, default_label,
                default_class, not_default_classes
            )

            df_dic[df_name] = df

        return df_dic

    @staticmethod
    def _fixup(
            df_name: str,
            df: pd.DataFrame,
            all_labels: List[str],
            default_label: str,
            default_class: str,
            not_default_classes: List[str]
    ) -> pd.DataFrame:
        """
        表現のデータフレームの調整

        Parameters
        ----------
        df_name : str
            調整対象データフレームの名前
        df : pd.DataFrame
            調整対象のデータフレーム
        all_labels : List[str]
            全ラベルのリスト
        default_label : str
            デフォルトラベル
        default_class : str
            デフォルトクラス
        not_default_classes : List[str]
            デフォルトではないクラスのリスト

        Returns
        -------
        pd.DataFrame
            調整済みのデータフレーム
        """
        cols_len = len(df.columns)
        df[cols_len] = None  # ※３

        df = df.apply(
            ExpressionDataframeDictionary._add_default_label,
            args=(default_label, df_name),
            axis=1
        )
        df.fillna(method='ffill', inplace=True)
        df = df.apply(
            ExpressionDataframeDictionary._stack_and_labeling,
            args=(all_labels,),
            axis=1
        )
        df.drop(columns = [col for col in range(3, cols_len + 1)], inplace=True)
        df = df.apply(
            ExpressionDataframeDictionary._classifying,
            args=(default_class, not_default_classes),
            axis=1
        )

        return df

    @staticmethod
    def _add_default_label(
            row: pd.Series, default_label: str, df_name: str
    ) -> pd.Series:
        """
        デフォルトラベルの追加

        全ての行で最後の有効な値を持つ列の次の列に、デフォルトラベルを追加する
        デフォルトではないラベルを持つ行にもデフォルトラベルが追加されるが、
        _stack_and_labelingで採用するラベルは、
        各行で最初に観測するラベルなので問題ない

        Parameters
        ----------
        row : pd.Series
            表現、表現のラベル、表現のクラスの情報を持つ行
            ラベルがデフォルトの行はラベルの情報を持たず、
            クラスがデフォルトの行はクラスの情報を持たない
            クラスを示す文字列は表現の中にある
        default_label : str
            デフォルトラベル
        df_name : str
            調整対象データフレームの名前

        Returns
        -------
        pd.Series
            デフォルトラベルが追加された行
        """
        last_valid_idx = row.last_valid_index()
        if last_valid_idx is None:
            print(f'{df_name}の{row.name}行目は空欄だが、デフォルトの行とする')
            label_idx = 0

        else:
            label_idx = last_valid_idx + 1

        row.values[label_idx] = default_label

        return row

    @staticmethod
    def _stack_and_labeling(row: pd.Series, all_labels: List[str]) -> pd.Series:
        """
        表現の列、ラベルの列へ集約

        fillnaされた行から、表現とラベルを抽出し、それぞれを一つのセルにまとめる

        Parameters
        ----------
        row : pd.Series
            表現、表現のラベル、表現のクラスの情報を持つ行
            全ての行がデフォルトのラベルを持ち、
            クラスがデフォルトの行はクラスの情報を持たない
            クラスを示す文字列は表現の中にある
        all_labels : List[str]
            全ラベルのリスト

        Returns
        -------
        pd.Series
            表現とラベルに分けられた行
        """
        for idx, value in enumerate(row.values):
            if value in all_labels:
                label = value

                row_valid_values = row.values[:idx]
                expr = ''.join(row_valid_values)

                break

        row.values[0] = expr
        row.values[1] = label
        row.values[2:] = None

        return row

    @staticmethod
    def _classifying(
            row: pd.Series, default_class: str, not_default_classes: List[str]
    ) -> pd.Series:
        """
        クラスの適用

        Parameters
        ----------
        row : pd.Series
            表現、表現のラベル、表現のクラスの情報を持つ行
            クラスがデフォルトの行はクラスの情報を持たない
            クラスを示す文字列は表現の中にある
        default_class : str
            デフォルトのクラス
        not_default_classes : List[str]
            デフォルトではないクラスのリスト

        Returns
        -------
        pd.Series
            表現、ラベル、クラスに分けられた行
        """
        class_label = default_class

        expr = row.values[0]
        for not_default_class in not_default_classes:
            if not_default_class in expr:
                expr = expr.replace(not_default_class, '')
                row.values[0] = expr

                class_label = not_default_class

                break

        row.values[2] = class_label

        return row


# # 実行

# In[4]:


read_path = '/content/drive/MyDrive/local_cuisine_search_app/data/raw_data/end_of_sentence.xlsx'
sheet_names = ['df1', 'df2', 'df3', 'df4']
file_name = 'end_of_sentences_dataframe'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list_dependencies/01_dataset_template_list/dataset_template_list_dependencies/02_end_of_sentences_dataframe'

df = DataframeMaker.create_and_save(read_path, sheet_names, file_name, save_dir)


# # 出力結果の確認

# In[5]:


df


# # メモ

# ※１
# - `.isin()`は引数のオブジェクト内のいずれかの要素を含む行をTrueにしたシリーズを返す

# ※２
# - `lst = list(dict.fromkeys(lst))`は、lst内の要素の順番はそのままに、重複をなくす処理

# ※３
# - 最後の列まで値がある行が、デフォルトのクラスだった場合、デフォルトのラベルを追加する列がないため、一列増やしている
