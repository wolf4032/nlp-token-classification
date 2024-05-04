from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Literal, Tuple

import gradio as gr

class GrComponent(ABC):
    """
    コンポーネントのクラス

    全てのコンポーネントを作成してから.render()できるようにするためのクラス
    各コンポーネントに対して実行される処理のメソッドを子クラスに追加していく

    Attributes
    ----------
    comp: gr.Component
        コンポーネント
    """
    def __init__(self, *param: Any):
        """
        コンストラクタ

        createメソッドでコンポーネントを作成する
        """
        self.comp: gr.Component = self._create(*param)
    
    @abstractmethod
    def _create(self, *param: Any) -> gr.Component:
        """
        コンポーネントの作成

        Returns
        -------
        gr.Component
            作成されるコンポーネント
        """
        pass


class GrLayout(ABC):
    """
    レイアウトのクラス

    gr.Rowや、gr.Columnを事前に定義しておくためのクラス

    Attributes
    ----------
    layout_type: gr.BlockContext, default None
        gr.Rowやgr.Columnなどの、子要素の配置スタイル
    children: (Dict[str, GrLayout | GrComponent] | List[GrLayout | GrComponent])
        子要素
    """
    layout_type: gr.BlockContext = None

    def __init__(self, *param: Any):
        """
        コンストラクタ

        子クラス固有のレイアウトタイプにオーバーライドされているか確認し、
        子要素をもつ辞書かリストを作成する

        Raises
        ------
        NotImplementedError
            レイアウトタイプがNoneの場合
        """
        if self.layout_type is None:
            raise NotImplementedError('layout_typeを設定してください')

        self.children: (
            Dict[str, GrLayout | GrComponent] | List[GrLayout | GrComponent]
        ) = self._create(*param)

    @abstractmethod
    def _create(
            self, *param: Any
    ) -> Dict[str, GrLayout | GrComponent] | List[GrLayout | GrComponent]:
        """
        子要素をまとめた辞書かリストの作成

        Returns
        -------
        Dict[str, GrLayout | GrComponent] | List[GrLayout | GrComponent]
            子要素が入った辞書かリスト
        """
        pass


class GrListener:
    """
    イベントリスナーのクラス

    GrBlocksで、まとめてイベントリスナーを設定できるように、
    必要な項目を定義しておくクラス
    本来、イベントリスナーはwith gr.Blocks():の中で定義するものだが、
    withの外で作成し、後でまとめてブロックに適用することができる
    """
    def __init__(
            self,
            trigger: Callable[[], Any] | List[Callable[[], Any]] = None,
            fn: Callable | None | Literal["decorator"] = None,
            inputs: GrComponent | list[GrComponent | GrLayout] | None = None,
            outputs: GrComponent | list[GrComponent | GrLayout] | None = None,
            api_name: str | None | Literal[False] = None,
            scroll_to_output: bool = False,
            show_progress: Literal["full", "minimal", "hidden"] = 'full',
            queue: bool | None = None,
            batch: bool = False,
            max_batch_size: int = 4,
            preprocess: bool = True,
            postprocess: bool = True,
            cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
            every: float | None = None,
            trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
            js: str | None = None,
            concurrency_limit: int | None | Literal["default"] = "default",
            concurrency_id: str | None = None,
            show_api: bool = True,
            thens: GrListener | List[GrListener] | None = None
    ):
        """
        コンストラクタ

        引数についてはGradioのソースコードを参照してください
        thenは、あるイベントリスナー実行直後に実行させる処理を指定するものです
        詳細は、Gradioの公式サイトを参照してください（https://www.gradio.app/guides/blocks-and-event-listeners#running-events-consecutively）
        triggerは、gr.onによる複数のトリガー指定に対応できるよう
        List[Callable[[], Any]]に対応している
        inputsとoutputsは、GrComponentのオブジェクトをそのまま指定するだけでよく、
        .compにする必要はない

        Raises
        ------
        NotImplementedError
            fnが設定されていない場合
        """
        if fn is None:
            raise NotImplementedError('fnを指定してください')

        self._trigger = trigger
        self.fn = fn
        self.inputs = InputsOutputsMaker.setup_inputs_or_outputs(inputs)
        self.outputs = InputsOutputsMaker.setup_inputs_or_outputs(outputs)
        self._api_name = api_name
        self._scroll_to_output = scroll_to_output
        self._show_progress = show_progress
        self._queue = queue
        self._batch = batch
        self._max_batch_size = max_batch_size
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._cancels = cancels
        self._every = every
        self._trigger_mode = trigger_mode
        self._js = js
        self._concurrency_limit = concurrency_limit
        self._concurrency_id = concurrency_id
        self._show_api = show_api
        self.thens = thens

    def setup(self) -> None:
        """
        イベントリスナーの適用

        GrBlocksの_set_event_listenerメソッドで実行されることで、
        イベントリスナーが適用される
        """
        if self.thens is None:
            self._setup_without_then()

        else:
            self._setup_with_then()

    def _setup_without_then(self) -> gr.events.Dependency:
        """
        thenがないイベントリスナーの適用

        Returns
        -------
        gr.events.Dependency
            イベントリスナー
        """
        if isinstance(self._trigger, list):
            dep = gr.on(
                triggers=self._trigger,
                fn=self.fn,
                inputs=self.inputs,
                outputs=self.outputs,
                api_name=self._api_name,
                scroll_to_output=self._scroll_to_output,
                show_progress=self._show_progress,
                queue=self._queue,
                batch=self._batch,
                max_batch_size=self._max_batch_size,
                preprocess=self._preprocess,
                postprocess=self._postprocess,
                cancels=self._cancels,
                every=self._every,
                trigger_mode=self._trigger_mode,
                js=self._js,
                concurrency_limit=self._concurrency_limit,
                concurrency_id=self._concurrency_id,
                show_api=self._show_api
            )

        else:
            dep = self._trigger(
                fn=self.fn,
                inputs=self.inputs,
                outputs=self.outputs,
                api_name=self._api_name,
                scroll_to_output=self._scroll_to_output,
                show_progress=self._show_progress,
                queue=self._queue,
                batch=self._batch,
                max_batch_size=self._max_batch_size,
                preprocess=self._preprocess,
                postprocess=self._postprocess,
                cancels=self._cancels,
                every=self._every,
                trigger_mode=self._trigger_mode,
                js=self._js,
                concurrency_limit=self._concurrency_limit,
                concurrency_id=self._concurrency_id,
                show_api=self._show_api
            )

        return dep

    def _setup_with_then(self) -> None:
        """
        thenがあるイベントリスナーの適用
        """
        dep = self._setup_without_then()

        all_thens = GrListener._d_f_s_thens(self.thens)

        for listener in all_thens:
            dep = listener.setup_then(dep)

    @staticmethod
    def _d_f_s_thens(
            thens: GrListener | List[GrListener], all_thens: List[GrListener] = []
    ) -> List[GrListener]:
        """
        全てのthenの取得

        thenの各要素が呼び出す全てのイベントを深さ優先探索で取得する
        入れ子の様になっている全てのイベントを最後まで呼び出せるようにする

        Parameters
        ----------
        thens : GrListener | List[GrListener]
            thensに代入されていたGrListenerか、そのリスト
        all_thens : List[GrListener], optional
            深さ優先探索で見つかったGrListenerが格納されるリスト, by default []

        Returns
        -------
        List[GrListener]
            深さ優先探索で見つかった全てのGrListenerが格納されているリスト
        """
        if isinstance(thens, list):
            for listener in thens:
                all_thens.append(listener)
                all_thens = GrListener._d_f_s_thens(listener.thens, all_thens)

        elif isinstance(thens, GrListener):
            all_thens.append(thens)

        return all_thens

    def setup_then(self, dep: gr.events.Dependency) -> gr.events.Dependency:
        """
        thenの追加

        イベントリスナーにthenを追加する

        Parameters
        ----------
        dep : gr.events.Dependency
            thenを追加されるイベントリスナー

        Returns
        -------
        gr.events.Dependency
            thenが追加されたイベントリスナー
        """
        dep.then(
            fn=self.fn,
            inputs=self.inputs,
            outputs=self.outputs,
            api_name=self._api_name,
            scroll_to_output=self._scroll_to_output,
            show_progress=self._show_progress,
            queue=self._queue,
            batch=self._batch,
            max_batch_size=self._max_batch_size,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
            cancels=self._cancels,
            every=self._every,
            trigger_mode=self._trigger_mode,
            js=self._js,
            concurrency_limit=self._concurrency_limit,
            concurrency_id=self._concurrency_id,
            show_api=self._show_api
        )

        return dep


class GrExamples:
    """
    gr.Componentへの入力例のクラス

    詳細は公式ページを参照してください（https://www.gradio.app/docs/examples/）
    """
    def __init__(
        self,
        examples: list[Any] | list[list[Any]] | str,
        inputs: GrComponent | list[GrComponent | GrLayout] | None = None,
        outputs: GrComponent | list[GrComponent | GrLayout] | None = None,
        fn: Callable | None = None,
        cache_examples: bool | Literal["lazy"] | None = None,
        examples_per_page: int = 10,
        label: str | None = "Examples",
        elem_id: str | None = None,
        run_on_click: bool = False,
        preprocess: bool = True,
        postprocess: bool = True,
        api_name: str | Literal[False] = "load_example",
        batch: bool = False,
        listener: GrListener | None = None
    ):
        """
        コンストラクタ

        Parameters（gr.Examplesにないもの）
        ----------
        listener: GrListener | None
            インスタンス化したGrListener
            これが渡されると、gr.Examplesのinputs,outputs,fnが、
            渡されたリスナーと同じものになる
        """
        self._examples = examples

        if listener is None:
            self._inputs = InputsOutputsMaker.setup_inputs_or_outputs(inputs)
            self._outputs = InputsOutputsMaker.setup_inputs_or_outputs(outputs)
            self._fn = fn
        else:
            self._inputs = listener.inputs
            self._outputs = listener.outputs
            self._fn = listener.fn

        self._cache_examples = cache_examples
        self._examples_per_page = examples_per_page
        self._label = label
        self._elem_id = elem_id
        self._run_on_click = run_on_click
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._api_name = api_name
        self._batch = batch

    def set_examples(self) -> None:
        """
        Examplesの適用
        """
        gr.Examples(
            examples=self._examples,
            inputs=self._inputs,
            outputs=self._outputs,
            fn=self._fn,
            cache_examples=self._cache_examples,
            examples_per_page=self._examples_per_page,
            label=self._label,
            elem_id=self._elem_id,
            run_on_click=self._run_on_click,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
            api_name=self._api_name,
            batch=self._batch
        )


class InputsOutputsMaker:
    """
    inputsとoutputs調整クラス

    GrListenerと、GrExamplesに渡されたinputsとoutputsを
    Gradioのイベントリスナーとgr.Examplesに渡せる形にするメソッドを持つクラス
    """
    @staticmethod
    def setup_inputs_or_outputs(
            in_or_outputs: GrComponent | GrLayout | list[GrComponent | GrLayout] | None
    ) -> gr.Component | List[gr.Component] | None:
        """
        inputsとoutputsをgr.Componentへ変換

        GrComponentから、gr.Componentである.compだけを抜き出す

        Parameters
        ----------
        in_or_outputs : GrComponent | list[GrComponent | GrLayout] | None
            イベントリスナーか、gr.Examplesのinputsかoutputs対象である
            gr.Componentを含むGrComponentや、GrComponentやGrLayoutのリスト

        Returns
        -------
        gr.Component | List[gr.Component] | None
            イベントリスナーかgr.Examplesの参照対象であるgr.Component
        """
        if in_or_outputs is None:
            return in_or_outputs

        else:
            if isinstance(in_or_outputs, GrComponent):
                return in_or_outputs.comp

            else:
                return InputsOutputsMaker._create_flat_comps(in_or_outputs)

    @staticmethod
    def _create_flat_comps(
            in_or_outputs: GrLayout | List[GrComponent | GrLayout]
    ) -> List[gr.Component]:
        """
        gr.Componentのリストの作成

        必要であればfind_compsを使って、gr.Componentのリストを作成する

        Parameters
        ----------
        in_or_outputs : List[GrComponent | GrLayout]
            イベントリスナーか、gr.Examplesのinputsかoutputs対象である
            gr.Componentを含むGrComponentやGrLayoutのリスト

        Returns
        -------
        List[gr.Component]
            イベントリスナーかgr.Examplesの参照対象であるgr.Componentのリスト
        """
        if isinstance(in_or_outputs, GrLayout):
            in_or_outputs = in_or_outputs.children

            if isinstance(in_or_outputs, dict):
                in_or_outputs = list(in_or_outputs.values())
            
        comps = []
        for obj in in_or_outputs:
            if isinstance(obj, GrComponent):
                comps.append(obj.comp)

            else:
                inner_flat_comps = find_comps(obj)
                comps.extend([comp.comp for comp in inner_flat_comps])

        return comps


class GrBlocks(ABC):
    """
    ブロックのクラス
    """
    @classmethod
    def create_and_launch(cls, *param: Any, debug: bool = False) -> gr.Blocks:
        """
        ブロックの作成と実行

        Parameters
        ----------
        debug : bool, optional, default False
            デバッグをするならTrue、しないならFalse
            デフォルトはFalse

        Returns
        -------
        gr.Blocks
            作成されたブロック
        """
        children, listeners = cls._create_children_and_listeners(*param)

        with gr.Blocks() as blocks:
            GrBlocks._render_children(children)

            GrBlocks._set_event_listener(listeners)

        blocks.launch(debug=debug)

        return blocks

    @classmethod
    @abstractmethod
    def _create_children_and_listeners(
            cls, *param: Any
    ) -> Tuple[Dict[str, Any] | List[Any], List[Any]]:
        """
        childrenとlistenersの作成
        
        Returns
        -------
        children : Dict[str, Any] | List[Any]
            ブロックの子要素をまとめた辞書かリスト
            内部の入れ子構造に決まりがないため、Anyとしているが、
            GrLayoutとGrComponentが多重のリストや辞書になったりならなかったりする
        listeners : List[Any]
            イベントリスナーがまとまったリスト
            内部の入れ子構造に決まりがないため、Anyとしているが、
            GrListenerが多重のリストになったりならなかったりする
        """
        pass

    @staticmethod
    def _render_children(children: Dict[str, Any] | List[Any]) -> None:
        """
        子要素のrender

        全ての子要素をまとめてrender()する
        コンポーネントをwith gr.Blocks():内で.render()することで、ブロックに実装される
        レイアウトはwith child.layout_type():をしてから子要素を.render()する

        Parameters
        ----------
        children : Dict[str, Any] | List[Any]
            子要素の辞書かリスト
        """
        if isinstance(children, dict):
            children = list(children.values())

        for child in children:
            if isinstance(child, GrLayout):
                with child.layout_type():
                    GrBlocks._render_children(child.children)

            elif isinstance(child, list):
                GrBlocks._render_children(child)

            elif isinstance(child, GrComponent):
                child.comp.render()

            elif isinstance(child, GrExamples):
                child.set_examples()

    @staticmethod
    def _set_event_listener(listeners: List[Any]) -> None:
        """
        イベントリスナーの適用

        全てのイベントリスナーをまとめて適用する

        Parameters
        ----------
        listeners : List[Any]
            イベントリスナーのリスト
        """
        for listener in listeners:
            if isinstance(listener, list):
                GrBlocks._set_event_listener(listener)

            else:
                listener.setup()

def find_comps(
        children: Dict[str, GrComponent | GrLayout] | GrLayout | List[Any],
        keys: List[str] | str |None = None
) -> List[GrComponent] | GrComponent:
    """
    コンポーネントの検索

    内包する全てのコンポーネントを検索し、
    flatten_compsを用いてフラットなコンポーネントのリストを作成する
    keysを渡すなら、childrenは辞書形式、keysを渡さないなら、childrenはGrLayout
    childrenがGrLayoutなら内包する全てのコンポーネントをリストにして返す
    同種のイベントリスナーをforで簡単に作成するために使う

    Parameters
    ----------
    children : Dict[str, GrComponent  |  GrLayout] | GrLayout | List[Any]
        子要素の辞書か、GrLayoutか、子要素のリスト
    keys : List[str] | str | None, optional, default None
        辞書のキーのリストか文字列
        辞書形式の子要素から、順番に深い子要素を取得していく

    Returns
    -------
    List[GrComponent] | GrComponent
        フラットなコンポーネントのリスト
    """
    if isinstance(keys, list):
        for key in keys:
            children: GrLayout | GrComponent = find_dict_value(children, key)

    elif isinstance(keys, str):
        children: GrLayout | GrComponent = find_dict_value(children, keys)

    else:
        children: List[GrComponent] = flatten_comps(children)

    return children

def find_dict_value(
        children: Dict[str, GrComponent | GrLayout], key: str
) -> GrLayout | GrComponent:
    """
    辞書のバリューの取得

    childrenのkeyのバリューが内包するGrLayoutかGrComponentを取得する

    Parameters
    ----------
    children : Dict[str, GrComponent  |  GrLayout]
        子要素の辞書
    key : str
        子要素の辞書から取得するバリューのキー

    Returns
    -------
    GrLayout | GrComponent
        取得したGrLayoutかGrComponent
    """
    if isinstance(children, dict):
        children: GrLayout | GrComponent = children[key]

    elif isinstance(children, GrLayout):
        children: GrLayout | GrComponent = children.children[key]

    return children

def flatten_comps(children: GrLayout | List[Any]) -> List[GrComponent]:
    """
    フラットなコンポーネントのリストの作成

    find_compsから呼び出される
    一つのGrLayout内で、同種のコンポーネントは
    リストに直接入れられている可能性があるため、それらも含めてフラットにする
    例：[comp1, comp2, [Comp(i) for i in range(10)], comp13]

    Parameters
    ----------
    children : GrLayout | List[Any]
        GrLayoutか子要素のリスト

    Returns
    -------
    List[GrComponent]
        フラットなコンポーネントのリスト
    """
    if isinstance(children, GrLayout):
        children = children.children

        if isinstance(children, dict):
            children = list(children.values())

    if isinstance(children, list):
        return [
            comp
            for grandchildren in children
            for comp in flatten_comps(grandchildren)
        ]

    else:
        return [children]

