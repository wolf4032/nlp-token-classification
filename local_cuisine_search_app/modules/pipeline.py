from typing import Dict, List

from transformers import BertJapaneseTokenizer, BertForTokenClassification, pipeline
from transformers.pipelines.token_classification import TokenClassificationPipeline

class NaturalLanguageProcessing:
    """
    固有表現を抽出するクラス

    model_dirにある言語モデルを使って固有表現抽出パイプラインを作成する
    モデルに固有表現を抽出させるメソッドを持つ

    Attributes
    ----------
    _nlp : TokenClassificationPipeline
        固有表現抽出パイプライン
    """
    def __init__(self, model_dir: str):
        """
        コンストラクタ

        _nlpを作成する

        Parameters
        ----------
        model_dir : str
            使用する言語モデルのディレクトリ
        """
        self._nlp = NaturalLanguageProcessing._create(model_dir)

    @staticmethod
    def _create(model_dir: str) -> TokenClassificationPipeline:
        """
        パイプラインの作成

        Parameters
        ----------
        model_dir : str
            使用する言語モデルのディレクトリ

        Returns
        -------
        TokenClassificationPipeline
            固有表現抽出パイプライン
        """
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
        model = BertForTokenClassification.from_pretrained(model_dir)

        nlp = pipeline(
            'token-classification',
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy='simple'
        )

        return nlp

    def classify(self, input: str) -> Dict[str, List[str]]:
        """
        固有表現の抽出

        Parameters
        ----------
        input : str
            固有表現抽出対象

        Returns
        -------
        Dict[str, List[str]]
            抽出結果の辞書
            キーが分類ラベル、バリューがそのラベルの文字列のリスト
        """
        prediction_results:List[Dict[str, str | float | None]] = self._nlp(input)

        classified_words = {}
        for predict_result in prediction_results:
            label = predict_result['entity_group']
            word = predict_result['word']

            if label not in classified_words:
                classified_words[label] = []

            classified_words[label].append(word.replace(' ', ''))

        return classified_words

    def classify_and_show(self, input: str) -> Dict[str, List[str]]:
        """
        固有表現の抽出と表示

        Parameters
        ----------
        input : str
            固有表現抽出対象

        Returns
        -------
        Dict[str, List[str]]
            抽出結果の辞書
            キーが分類ラベル、バリューがそのラベルの文字列のリスト
        """
        classified_words = self.classify(input)

        for label, words in classified_words.items():
            print(f'{label: <10} {"、".join(words)}')

        return classified_words

