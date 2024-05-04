#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from typing import Dict

import sys
sys.path.append('/content/drive/MyDrive/local_cuisine_search_app/modules')

from utility import dump_obj_as_json


# # 実行

# In[4]:


def create_and_save(file_name: str, save_dir: str) -> Dict[str, Dict[str, str]]:
    area_dic = {
        '北海道': '北海道',
        '青森': '青森県',
        '岩手': '岩手県',
        '宮城': '宮城県',
        '秋田': '秋田県',
        '山形': '山形県',
        '福島': '福島県',
        '茨城': '茨城県',
        '栃木': '栃木県',
        '群馬': '群馬県',
        '埼玉': '埼玉県',
        '千葉': '千葉県',
        '東京': '東京都',
        '神奈川': '神奈川県',
        '山梨': '山梨県',
        '長野': '長野県',
        '静岡': '静岡県',
        '新潟': '新潟県',
        '富山': '富山県',
        '石川': '石川県',
        '福井': '福井県',
        '岐阜': '岐阜県',
        '愛知': '愛知県',
        '三重': '三重県',
        '滋賀': '滋賀県',
        '京都': '京都府',
        '大阪': '大阪府',
        '兵庫': '兵庫県',
        '奈良': '奈良県',
        '和歌山': '和歌山県',
        '鳥取': '鳥取県',
        '島根': '島根県',
        '岡山': '岡山県',
        '広島': '広島県',
        '山口': '山口県',
        '徳島': '徳島県',
        '香川': '香川県',
        '愛媛': '愛媛県',
        '高知': '高知県',
        '福岡': '福岡県',
        '佐賀': '佐賀県',
        '長崎': '長崎県',
        '熊本': '熊本県',
        '大分': '大分県',
        '宮崎': '宮崎県',
        '鹿児島': '鹿児島県',
        '沖縄': '沖縄県',
        '北海道地方': '北海道',
        '東北地方': '東北',
        '北東北地方': '北東北',
        '南東北地方': '南東北',
        '関東地方': '関東',
        '北関東地方': '北関東',
        '南関東地方': '南関東',
        '中部地方': '中部',
        '甲信越地方': '甲信越',
        '北陸地方': '北陸',
        '東海地方': '東海',
        '近畿地方': '近畿',
        '関西地方': '関西',
        '中国地方': '中国',
        '四国地方': '四国',
        '九州地方': '九州',
        '沖縄地方': '沖縄'
    }

    type_dic = {
        '飯系': '飯料理',
        '米系': '飯料理',
        '米料理': '飯料理',
        'お米料理': '飯料理',
        'ご飯もの': '飯料理',
        '肉系': '肉料理',
        'お肉料理': '肉料理',
        '野菜系': '野菜料理',
        'お野菜料理': '野菜料理',
        '魚系': '魚料理',
        'お魚料理': '魚料理'
    }

    ingr_dic = {
        'ご飯': '米',
        'ごはん': '米',
        '白飯': '米',
        '精白米': '米',
        'お米': '米',
        '麦飯': '麦',
        'すし飯': '酢飯',
        '餅米': 'もち米',
        '鶏': '鶏肉',
        'とり肉': '鶏肉',
        'トリ肉': '鶏肉',
        'チキン': '鶏肉',
        '豚': '豚肉',
        'ぶた肉': '豚肉',
        'ブタ肉': '豚肉',
        'ポーク': '豚肉',
        '牛': '牛肉',
        'ビーフ': '牛肉',
        '猪': '猪肉',
        'いのしし': '猪肉',
        'イノシシ': '猪肉',
        'いのしし肉': '猪肉',
        'イノシシ肉': '猪肉',
        'くじら': '鯨肉',
        'クジラ': '鯨肉',
        '鯨': '鯨肉',
        'くじら肉': '鯨肉',
        'クジラ肉': '鯨肉',
        'にんじん': '人参',
        'ニンジン': '人参',
        'ごぼう': 'ゴボウ',
        '牛蒡': 'ゴボウ',
        'だいこん': '大根',
        'ダイコン': '大根',
        'ながねぎ': 'ねぎ',
        'しろねぎ': 'ねぎ',
        'あおねぎ': 'ねぎ',
        'ネギ': 'ねぎ',
        'ナガネギ': 'ねぎ',
        'シロネギ': 'ねぎ',
        'アオネギ': 'ねぎ',
        '葱': 'ねぎ',
        '長ねぎ': 'ねぎ',
        '長ネギ': 'ねぎ',
        '長葱': 'ねぎ',
        '白ねぎ': 'ねぎ',
        '白ネギ': 'ねぎ',
        '白葱': 'ねぎ',
        '青ねぎ': 'ねぎ',
        '青ネギ': 'ねぎ',
        '青葱': 'ねぎ',
        '生しいたけ': 'しいたけ',
        '干ししいたけ': 'しいたけ',
        'シイタケ': 'しいたけ',
        '生シイタケ': 'しいたけ',
        '干しシイタケ': 'しいたけ',
        '椎茸': 'しいたけ',
        '生椎茸': 'しいたけ',
        '干し椎茸': 'しいたけ',
        'さといも': '里芋',
        'サトイモ': '里芋',
        'れんこん': 'レンコン',
        '蓮根': 'レンコン',
        'なす': 'ナス',
        '茄子': 'ナス',
        'たけのこ': 'タケノコ',
        '竹の子': 'タケノコ',
        '筍': 'タケノコ',
        'きゅうり': 'キュウリ',
        '胡瓜': 'キュウリ',
        'はくさい': '白菜',
        'ハクサイ': '白菜',
        'じゃがいも': 'ジャガイモ',
        'じゃが芋': 'ジャガイモ',
        'たまねぎ': '玉ねぎ',
        'タマネギ': '玉ねぎ',
        '玉ネギ': '玉ねぎ',
        '玉葱': '玉ねぎ',
        'サツマイモ': 'さつまいも',
        'さつま芋': 'さつまいも',
        '薩摩芋': 'さつまいも',
        'かぶ': 'カブ',
        '蕪': 'カブ',
        'かぼちゃ': 'カボチャ',
        '南瓜': 'カボチャ',
        'サヤインゲン': 'さやいんげん',
        'きのこ': 'キノコ',
        '茸': 'キノコ',
        'しゅんぎく': '春菊',
        'シュンギク': '春菊',
        'ほうれんそう': 'ほうれん草',
        'ホウレンソウ': 'ほうれん草',
        'ホウレン草': 'ほうれん草',
        '法蓮草': 'ほうれん草',
        'わさび': 'ワサビ',
        '山葵': 'わさび',
        'とうもろこし': 'トウモロコシ',
        '玉蜀黍': 'トウモロコシ',
        'きくらげ': 'キクラゲ',
        '木耳': 'キクラゲ',
        'いんげんまめ': 'いんげん',
        'インゲン': 'いんげん',
        'インゲンマメ': 'いんげん',
        'いんげん豆': 'いんげん',
        'インゲン豆': 'いんげん',
        'きゃべつ': 'キャベツ',
        'くり': '栗',
        'クリ': '栗',
        '小豆': 'あずき',
        'みょうが': 'ミョウガ',
        '茗荷': 'ミョウガ',
        'そらまめ': 'ソラマメ',
        'そら豆': 'ソラマメ',
        'まつたけ': 'マツタケ',
        '松茸': 'マツタケ',
        'まいたけ': 'マイタケ',
        '舞茸': 'マイタケ',
        'モヤシ': 'もやし',
        'サヤエンドウ': 'さやえんどう',
        'えんどう': 'エンドウ',
        'やまいも': '山芋',
        'ヤマイモ': '山芋',
        'へちま': 'ヘチマ',
        '糸瓜': 'ヘチマ',
        'ふき': 'フキ',
        '蕗': 'フキ',
        'にら': 'ニラ',
        '韮': 'ニラ',
        'わらび': 'ワラビ',
        '蕨': 'ワラビ',
        'ゴーヤー': 'ゴーヤ',
        'にがうり': 'ゴーヤ',
        'ニガウリ': 'ゴーヤ',
        '苦瓜': 'ゴーヤ',
        'えだまめ': '枝豆',
        'エダマメ': '枝豆',
        'さんしょう': '山椒',
        'サンショウ': '山椒',
        'えのき': 'エノキ',
        '榎': 'エノキ',
        'シメジ': 'しめじ',
        '湿地': 'しめじ',
        'ぎんなん': '銀杏',
        'ギンナン': '銀杏',
        'とうがん': '冬瓜',
        'トウガン': '冬瓜',
        'さば': 'サバ',
        '鯖': 'サバ',
        'さんま': 'サンマ',
        '秋刀魚': 'サンマ',
        'あじ': 'アジ',
        '鯵': 'アジ',
        'さけ': 'サケ',
        '鮭': 'サケ',
        'いわし': 'イワシ',
        '鰯': 'イワシ',
        'まいわし': 'マイワシ',
        'かたくちいわし': 'カタクチイワシ',
        'にしん': 'ニシン',
        '鰊': 'ニシン',
        'まぐろ': 'マグロ',
        '鮪': 'マグロ',
        'さわら': 'サワラ',
        '鰆': 'サワラ',
        'かつお': 'カツオ',
        '鰹': 'カツオ',
        'たい': 'タイ',
        '鯛': 'タイ',
        '真鯛': 'マダイ',
        'たら': 'タラ',
        'まだら': 'タラ',
        'マダラ': 'タラ',
        '鱈': 'タラ',
        '真鱈': 'タラ',
        'すけとうだら': 'スケトウダラ',
        'すけそうだら': 'スケトウダラ',
        'スケソウダラ': 'スケトウダラ',
        '助惣鱈': 'スケトウダラ',
        'ひらめ': 'ヒラメ',
        '鮃': 'ヒラメ',
        '平目': 'ヒラメ',
        'かれい': 'カレイ',
        '鰈': 'カレイ',
        'すずき': 'スズキ',
        '鱸': 'スズキ',
        'ぶり': 'ブリ',
        '鰤': 'ブリ',
        'ふな': 'フナ',
        '鮒': 'フナ',
        'はたはた': 'ハタハタ',
        '鰰': 'ハタハタ',
        'うなぎ': 'ウナギ',
        '鰻': 'ウナギ',
        'さめ': 'サメ',
        '鮫': 'サメ',
        'わかさぎ': 'ワカサギ',
        '若鷺': 'ワカサギ',
        'しらす': 'シラス',
        'あゆ': 'アユ',
        '鮎': 'アユ',
        'あなご': 'アナゴ',
        '穴子': 'アナゴ',
        'たこ': 'タコ',
        '蛸': 'タコ',
        'えび': 'エビ',
        '海老': 'エビ',
        'いか': 'イカ',
        '烏賊': 'イカ',
        'ほたて': 'ホタテ',
        '帆立': 'ホタテ',
        'はまぐり': 'ハマグリ',
        '蛤': 'ハマグリ',
        'あわび': 'アワビ',
        '鮑': 'アワビ',
        'あさり': 'アサリ',
        '浅蜊': 'アサリ',
        'しじみ': 'シジミ',
        '蜆': 'シジミ',
        'ちりめん': 'ちりめんじゃこ',
        'じゃこ': 'ちりめんじゃこ',
        '縮緬': 'ちりめんじゃこ',
        '縮緬雑魚': 'ちりめんじゃこ',
        'かます': 'カマス',
        'ます': 'マス',
        '鱒': 'マス',
        'いくら': 'イクラ',
        'きんめだい': 'キンメダイ',
        '金目鯛': 'キンメダイ',
        'このしろ': 'コノシロ',
        '鮗': 'コノシロ',
        'かずのこ': 'カズノコ',
        '数の子': 'カズノコ',
        'さざえ': 'サザエ',
        '栄螺': 'サザエ',
        'こはだ': 'コハダ',
        'きす': 'キス',
        '鱚': 'キス',
        'ほっき': 'ホッキ',
        'ほっき貝': 'ホッキ',
        'ホッキ貝': 'ホッキ',
        'たちうお': '太刀魚',
        'タチウオ': '太刀魚',
        'まだこ': '真蛸',
        'マダコ': '真蛸',
        'かじき': 'カジキ',
        'しいら': 'シイラ',
        'とびうお': 'トビウオ',
        '飛魚': 'トビウオ',
        '飛び魚': 'トビウオ',
        'めだい': 'メダイ',
        '目鯛': 'メダイ',
        'いさき': 'イサキ',
        'かんぱち': 'カンパチ',
        '間八': 'カンパチ',
        'かに': 'カニ',
        '蟹': 'カニ',
        'ほや': 'ホヤ',
        'ひらまさ': 'ヒラマサ',
        '平鰤': 'ヒラマサ',
        '平政': 'ヒラマサ',
        'はも': 'ハモ',
        '鱧': 'ハモ',
        'するめ': 'スルメ',
        '鯣': 'スルメ',
        'しらうお': '白魚',
        'シラウオ': '白魚',
        'どじょう': 'ドジョウ',
        '鰌': 'ドジョウ',
        '鯲': 'ドジョウ',
        '泥鰌': 'ドジョウ',
        'こい': '鯉',
        'コイ': '鯉',
        'しらこ': '白子',
        'シラコ': '白子',
        'いわな': 'イワナ',
        '岩魚': 'イワナ',
        '嘉魚': 'イワナ',
        'たらこ': 'タラコ',
        '鱈子': 'タラコ',
        'うつぼ': 'ウツボ',
        'はぜ': 'ハゼ',
        '鯊': 'ハゼ',
        '沙魚': 'ハゼ',
        '蝦虎': 'ハゼ',
        '蝦虎魚': 'ハゼ',
        'あんこう': 'アンコウ',
        '鮟鱇': 'アンコウ',
        'ふぐ': 'フグ',
        '河豚': 'フグ',
        'なまず': 'ナマズ',
        '鯰': 'ナマズ',
        '鱠': 'ナマズ',
        'ぼら': 'ボラ',
        '鯔': 'ボラ',
        'えい': 'エイ',
        '海鷂魚': 'エイ',
        'ししゃも': 'シシャモ',
        '柳葉魚': 'シシャモ',
        'さとう': '砂糖',
        'サトウ': '砂糖',
        'しお': '塩',
        'シオ': '塩',
        'す': '酢',
        'ス': '酢',
        'しょうゆ': '醤油',
        'ショウユ': '醤油',
        'みそ': '味噌',
        'ミソ': '味噌',
        'ミリン': 'みりん',
        'だし': '出汁',
        'ダシ': '出汁',
        'だし汁': '出汁',
        'ごまあぶら': 'ごま油',
        'ゴマアブラ': 'ごま油',
        'ゴマ油': 'ごま油',
        '胡麻油': 'ごま油',
        'ゴマ': 'ごま',
        '胡麻': 'ごま',
        'かつおぶし': 'カツオ節',
        'かつお節': 'カツオ節',
        '鰹節': 'カツオ節',
        'しょうが': 'ショウガ',
        '生姜': 'ショウガ',
        'きなこ': 'きな粉',
        '黄な粉': 'きな粉',
        'ワサビ': 'わさび',
        'からし': 'カラシ',
        '辛子': 'カラシ',
        '練りからし': 'カラシ',
        '練り辛子': 'カラシ'
    }

    unify_dic = {
        'AREA': area_dic,
        'TYPE': type_dic,
        'INGR': ingr_dic
    }

    dump_obj_as_json(unify_dic, file_name, save_dir)

    return unify_dic


# # 実行

# In[5]:


file_name = 'unifying_dictionaries'
save_dir = '/content/drive/MyDrive/local_cuisine_search_app/data/processed_data/01_unifying_dictionaries'

unify_dic = create_and_save(file_name, save_dir)


# # 出力結果の確認

# In[6]:


unify_dic
