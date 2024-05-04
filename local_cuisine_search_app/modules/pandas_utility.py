import pandas as pd
import numpy as np
import ast

from abc import ABC, abstractmethod
from typing import List, Any

def read_csv_df(read_path: str, header: int = 0) -> pd.DataFrame:
    """
    csv形式のデータフレームの読み込み

    文字列として読み込まれたリストの復元も行う

    Parameters
    ----------
    read_path : str
        データフレームが保存されているパス
    header : int, optional
        ヘッダーとする行の指定, by default 0

    Returns
    -------
    pd.DataFrame
        データフレーム
    """
    df = pd.read_csv(read_path, header=header)

    lst_col_names = df.columns[df.iloc[0].apply(check_lst_col)]

    df[lst_col_names] = df[lst_col_names].applymap(ast.literal_eval)

    return df

def check_lst_col(value: str) -> bool:
    """
    リストの列の確認

    元の値がリストだった列の確認

    Parameters
    ----------
    value : str
        その列の代表の値

    Returns
    -------
    bool
        その列の値がリストならTrue、リストでなければFalse
    """
    # 数値に、インデックスで最初の文字を指定するとエラーになる
    is_str = isinstance(value, str)

    # 空文字に、インデックスで最初の文字を指定するとエラーになる
    is_not_empty = len(value) > 0

    is_lst = value[0] == '['

    is_lst_col = is_str and is_not_empty and is_lst

    return is_lst_col

def save_csv_df(df: pd.DataFrame, file_name: str, save_dir: str) -> None:
    """
    csv形式でのデータフレームの保存

    Parameters
    ----------
    df : pd.DataFrame
        保存するデータフレーム
    file_name : str
        保存するデータフレームのファイル名
    save_dir : str
        データフレームの保存先ディレクトリ
    """
    df.to_csv(f'{save_dir}/{file_name}.csv', index=False)

class CustomError(Exception, ABC):
    """
    カスタムエラーの基底クラス
    """
    def __init__(self, occurred_error: Exception, *message_args: Any):
        """
        CustomErrorのコンストラクタ

        Parameters
        ----------
        occurred_error : Exception
            発生したエラー
        """
        self.message_first_line = f'エラーが発生しました: {occurred_error}'
        self.message = self.create(*message_args)
        super().__init__(self.message)

    @abstractmethod
    def create(self, *message_args: Any) -> str:
        """
        エラーメッセージを作成するメソッド

        Returns
        -------
        str
            エラーメッセージ
        """
        pass

    def create_multi_line_message(self, lines: List[str]) -> str:
        """
        規定の文字列を先頭に付けて、エラーメッセージを改行させるメソッド

        Parameters
        ----------
        lines : List[str]
            エラーメッセージの各行の内容を持つリスト

        Returns
        -------
        str
            エラーメッセージ
        """
        lines = [self.message_first_line] + lines

        message = '\n\t'.join(lines)

        return message


class ColumnNameError(CustomError):
    """
    存在しない列名を渡した際に起きるカスタムエラーのクラス
    """
    def create(self, col_name: str) -> str:
        """
        エラーメッセージを作成するメソッド

        Parameters
        ----------
        col_name : str
            存在しなかった列名

        Returns
        -------
        str
            エラーメッセージ
        """
        lines = [
            f'"{col_name}"という列は見つかりませんでした',
            '列名を変更するか、番号で列を指定してください'
        ]

        message = self.create_multi_line_message(lines)

        return message


class DataframeIndexError(CustomError):
    """
    存在しないインデックスを指定した際に起きるカスタムエラーのクラス
    """
    def create(self, df: pd.DataFrame) -> str:
        """
        エラーメッセージを作成するメソッド

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム

        Returns
        -------
        str
            エラーメッセージ
        """
        rows_len = len(df)
        cols_len = len(df.columns)

        lines = [
            f'このデータフレームは、{rows_len}行{cols_len}列です',
            f'row_idxは{rows_len - 1}、colは{cols_len - 1}以下にしてください'
        ]

        message = self.create_multi_line_message(lines)

        return message


def get_col_idx(df: pd.DataFrame, col_name: str) -> int:
    """
    列名から、その列のインデックスを取得するメソッド

    Parameters
    ----------
    df : pd.DataFrame
        インデックスを取得するデータフレーム
    col_name : str
        インデックスを取得する列名

    Returns
    -------
    int
        列のインデックス

    Raises
    ------
    ColumnNameError
        存在しない列名
    """
    try:
        col_idx = df.columns.values.tolist().index(col_name)

    except ValueError as e:
        raise ColumnNameError(e, col_name)

    return col_idx


def update_row(
        df: pd.DataFrame, row_idx: int, col_idx_or_name: int | str,
        value: Any) -> None:
    """
    指定した一つのセルの値を更新するメソッド

    Parameters
    ----------
    df : pd.DataFrame
        更新対象のデータフレーム
    row_idx : int
        更新するセルの行インデックス
    col : int | str
        更新するセルの列インデックスか、列名
    value : Any
        セルに入れる値

    Raises
    ------
    ColumnNameError
        存在しない列名
    DataframeIndexError
        存在しないインデックス
    """

    if isinstance(col_idx_or_name, str):
        col_name = col_idx_or_name

        if col_name not in df.columns:
            raise ColumnNameError('カスタムエラー', col_name)

        df.at[row_idx, col_name] = value

    else:
        col_idx = col_idx_or_name

        try:
            df.iat[row_idx, col_idx] = value

        except IndexError as e:
            raise DataframeIndexError(e, df)

