from typing import Iterable, Any, Dict, Tuple, List
import json
import random

def load_json_obj(load_path: str) -> Any:
    """
    jsonファイルの読み込み

    Parameters
    ----------
    load_path : str
        ファイルが保存されているパス

    Returns
    -------
    Any
        読み込んだオブジェクト
    """
    with open(load_path, 'r')as f:
        obj: Any = json.load(f)

    return obj

def dump_obj_as_json(obj: Any, file_name: str, dump_dir: str) -> None:
    """
    jsonファイルの保存

    Parameters
    ----------
    obj : Any
        保存するオブジェクト
    file_name : str
        保存するファイル名
    dump_dir : str
        保存先のディレクトリ
    """
    with open(f'{dump_dir}/{file_name}.json', 'w') as f:
        json.dump(obj, f)

def search_strs(string: str, target_strs: Iterable[str]) -> List[str]:
    """
    文字列内の特定の要素の検出

    string内に含まれるtarget_strsの要素を、string内での出現順にリストにまとめる

    Parameters
    ----------
    string : str
        含まれている要素を検索される文字列
    target_strs : Iterable[str]
        検索対象の文字列のイテラブルオブジェクト

    Returns
    -------
    List[str]
        検出した要素が出現順に入れられたリスト
    """
    replace_str_dic: Dict[str, str] = {
        s_str: '#' * len(s_str) for s_str in target_strs
    }
    idxs_dic: Dict[int, str] = {}

    should_search: bool = True

    while should_search:
        should_search = False

        for s_str in target_strs:
            idx: int = string.find(s_str)

            if idx != -1:
                if not should_search:
                    should_search = True

                idxs_dic[idx] = s_str

                replace_str: str = replace_str_dic[s_str]
                string: str = string.replace(s_str, replace_str, 1)

    idxs: List[int] = list(idxs_dic.keys())
    idxs.sort()

    searched_strs: List[str] = [idxs_dic[idx] for idx in idxs]

    return searched_strs

def remove_extra_item(
        iterable_obj: Iterable[Any], extra_item: Any
) -> Iterable[Any]:
    """
    余分な要素の削除

    イテラブルオブジェクトから、余分な要素を削除する

    Parameters
    ----------
    iterable_obj : Iterable[Any]
        余分な要素を含むイテラブルオブジェクト
    extra_item : Any
        削除対象の要素

    Returns
    -------
    Iterable[Any]
        余分な要素を削除されたイテラブルオブジェクト
    """
    iterable_obj: Iterable[Any] = [obj for obj in iterable_obj if obj != extra_item]

    return iterable_obj

class RandomPicker:
    """
    ランダムに抽出するクラス

    処理上の抽出対象を要素のインデックスにすることで、
    全ての要素が抽出された際に、簡単に一から参照しなおせるようにした
        一度抽出された要素は、全ての要素が抽出され終わるまで再度抽出されない

    Attributes
    ----------
    _dic : Dict[str, Tuple[Any] | int | List[int]]
        抽出対象のタプルと、要素の総数の数値と、
        各要素のインデックスのリストをバリューにもつ辞書
    """
    def __init__(self, iterable_obj: Iterable[Any]):
        """
        コンストラクタ

        _dicを作成する

        Parameters
        ----------
        iterable_obj : Iterable[Any]
            抽出対象のイテラブルオブジェクト
        """
        self._dic: Dict[str, Tuple[Any] | int | List[int]]
        self._dic = self._create(iterable_obj)

    def _create(
            self, iterable_obj: Iterable[Any]
    ) -> Dict[str, Tuple[Any] | int | List[int]]:
        """
        _dicの作成

        Parameters
        ----------
        iterable_obj : Iterable[Any]
            抽出対象のイテラブルオブジェクト

        Returns
        -------
        Dict[str, Tuple[Any] | int | List[int]]
            抽出対象のタプルと、要素の総数の数値と、
            各要素のインデックスのリストをバリューに持つ辞書
        """
        self._is_iterable(iterable_obj)

        choices: Tuple[Any] = tuple(iterable_obj)
        length: int = len(choices)

        dic: Dict[str, Tuple[Any] | int] = {
            'choices': choices, 'length': length
        }
        self._fill_idxs(dic)

        return dic

    def _is_iterable(self, iterable_obj: Iterable[Any]) -> None:
        """
        イテラブルオブジェクトの判別

        渡されたオブジェクトがイテラブルかどうか判別する

        Parameters
        ----------
        iterable_obj : Iterable[Any]
            抽出対象のイテラブルオブジェクト

        Raises
        ------
        TypeError
            渡されたオブジェクトがイテラブルでなかった場合
        """
        try:
            iter(iterable_obj)

        except TypeError:
            raise TypeError(
                'iterable_objには、イテラブルなオブジェクトを入力してください'
            )

    def _fill_idxs(
            self, dic: Dict[str, Tuple[Any] | int] | None = None
    ) -> None:
        """
        インデックスのリストの作成

        インデックスのリストを作成する
        _dicの初期化時と、self.pick()で
        インデックスのリストが空になったときに実行される

        Parameters
        ----------
        dic : Dict[str, Tuple[Any]  |  int] | None, optional
            抽出対象のタプルと、要素の総数の数値をバリューに持つ辞書
            by default None
        """
        if dic is None:
            dic: Dict[str, Tuple[Any] | int | List[int]] = self._dic

        length: int = dic['length']

        idxs: List[int] = [idx for idx in range(length)]

        dic['idxs'] = idxs

    def pick(self) -> Any:
        """
        要素の抽出

        ランダムに要素を抽出する
        抽出対象を一周したら要素（のインデックス）を補充する

        Returns
        -------
        Any
            抽出した要素
        """
        idxs: List[int] = self._dic['idxs']

        choosen_idx: int = random.choice(idxs)

        idxs.remove(choosen_idx)

        if not idxs:
            self._fill_idxs()

        choosen_item: Any = self._dic['choices'][choosen_idx]

        return choosen_item

