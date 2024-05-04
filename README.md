# nlp-token-classification
ファインチューニングで、固有表現抽出用の言語モデルを作成しました

## 概要
例えば、「東京の肉料理で、春に食べられる、鶏肉を使った料理を教えてください」という文章を入力すると、
「東京　→　都道府県/地方」　「肉料理　→　種類」　「春　→　季節」　「鶏肉　→　食材」 のように、
固有表現を抽出する言語モデルを作成しました

## 各種リンク

### デモアプリ
[wolf4032/japanese-token-classification-search-local-cuisine](https://huggingface.co/spaces/wolf4032/japanese-token-classification-search-local-cuisine)
- 入力文から抽出された固有表現をもとに、日本の郷土料理を検索するアプリ
- [うちの郷土料理：農林水産省](https://www.maff.go.jp/j/keikaku/syokubunka/k_ryouri/index.html)
  - こちらのサイトに掲載されている、飯料理、肉料理、野菜料理、魚料理を検索します

### 言語モデル
[wolf4032/bert-japanese-token-classification-search-local-cuisine](https://huggingface.co/wolf4032/bert-japanese-token-classification-search-local-cuisine)
- ファインチューニングした言語モデル
- [使用させていただいた事前学習済みモデル](https://huggingface.co/tohoku-nlp/bert-base-japanese-v2)(tohoku-nlp/bert-base-japanese-v2)
- テスト結果
  - f1: 0.9961977186311787
  - accuracy: 0.9995689655172414
  - precision: 0.9940978077571669
  - recall: 0.9983065198983911

### モデルの学習に使ったデータセット
[wolf4032/token-classification-japanese-search-local-cuisine](https://huggingface.co/datasets/wolf4032/token-classification-japanese-search-local-cuisine)
- 自作のデータセット
- [このノートブックで作成した](local_cuisine_search_app/notebooks/create_data/04_encoded_dataset_dataframe/encoded_dataset_dataframe_dependencies/01_untokenized_dataset_list/untokenized_dataset_list.ipynb)
- ↓データの構造
```python
{
    'text': '関西地方あるいは四国地方の、秋に食べられているしいらを使用した魚料理があったら、検索。',
    'entities': [
        {
            'name': '関西地方',
            'span': [0, 4],
            'type': 'AREA'
        },
        {
            'name': '四国地方',
            'span': [8, 12],
            'type': 'AREA'
        },
        {
            'name': '秋',
            'span': [14, 15],
            'type': 'SZN'
        },
        {
            'name': 'しいら',
            'span': [23, 26],
            'type': 'INGR'
        },
        {
            'name': '魚料理',
            'span': [31, 34],
            'type': 'TYPE'
        }
    ]
}
```
