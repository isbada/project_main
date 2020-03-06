'''
第1个表格处理函数
# # vader score情感检测
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# analyzer = SentimentIntensityAnalyzer()
# vs = analyzer.polarity_scores(no_hashtag_text)
'''

import pandas as pd
import numpy as np
import re
from pattern.en import sentiment, positive, mood, modality
import spacy
from tqdm import tqdm
from nltk.tokenize import sent_tokenize  # 英文分句
from google_vision_api import get_labelScore_by_api

import os
# from igramscraper.instagram import Instagram


# ------------------------全局变量-------------------------------


p_emoji = re.compile('[\U00010000-\U0010ffff]')

p_hashtag = re.compile('#\w+')

nlp = spacy.load("en_core_web_sm")

p_img_name = re.compile('([^/]*?)\?')

# # intgram爬虫工具
# instagram = Instagram()
# # authentication supported
# instagram.with_credentials('qianqianandtree@gmail.com', 'erlenda1')
# instagram.login()


# ------------------------辅助函数-------------------------------


# ------------------------第2个表格处理-------------------------------


# 读取原始表格
df1 = pd.read_csv('第一阶段结果.csv', lineterminator='\n')


# 根据字段过滤不要的数据
mask = df1['Text_Include_Url'] | df1['Text_Include_Tel'] | df1['Text_Repetition'] | df1['Img_Dominant_Color_Difference'] | (
    df1['Text_Include_Stopwords'] != 'False')


# 丢弃不需要的列
df2 = df1[~mask].drop(['Text_Include_Url', 'Text_Include_Tel', 'Text_Include_Stopwords',
                       'Text_Repetition', 'Img_Dominant_Color_Difference'], axis=1)

# =============== 1.1 文本处理 ===============


Text_Emoji = []
Text_Emoji_Number = []
Text_Hashtags = []
Text_NoHashtag = []
Text_NoHashtag_Length = []
# Text_NoHashtag_Sentiment_assessments = []
Text_NoHashtag_Sentiment_sentiment = []
Text_NoHashtag_Sentiment_subjectivity = []
Text_NoHashtag_Sentiment_sentiment_abs, Text_NoHashtag_Sentiment_subjectivity_abs = [], []
Text_NoHashtag_Sentiment_positive = []
# Text_NoHashtag_Sentiment_mood = []
# Text_NoHashtag_Sentiment_modality = []
Text_NounPhrases = []
Text_Verbs = []

for idx, text in tqdm(enumerate(df2.Text, start=1), desc='2.1文本处理', total=len(df2)):
    # emoji检测
    emoji_li = p_emoji.findall(text)
    emoji_text = ' '.join(emoji_li)
    emoji_N = len(emoji_li)

    Text_Emoji.append(emoji_text)
    Text_Emoji_Number.append(emoji_N)
    # if emoji_li:
    #     print(idx, emoji_li)

    # hashtag检测,去掉hashtag,情感检测,名词动词情感词提取
    hashtag_li = p_hashtag.findall(text)
    hashtag_text = ';'.join(hashtag_li)
    no_hashtag_text = p_hashtag.sub('', text).strip()
    no_hashtag_len = len(re.split('\s+', no_hashtag_text))

    Text_Hashtags.append(hashtag_text)
    Text_NoHashtag.append(no_hashtag_text)
    Text_NoHashtag_Length.append(no_hashtag_len)

    # if hashtag_text:
    #     print(idx, hashtag_text)
    #     print(idx, no_hashtag_text, no_hashtag_len)

    # 情感检测: pattern库

    # this_assessments = sentiment(no_hashtag_text).assessments
    # this_sentiment = sentiment(no_hashtag_text)

    # 分句后求情感值和主观值(累加)
    sentence_li = sent_tokenize(no_hashtag_text)  # 段落分句
    val_li = np.array([sentiment(sentence) for sentence in sentence_li])
    try:
        this_sentiment, this_subjectivity = val_li.sum(0)
        # sentiment subject 绝对值运算
        this_sentiment_abs, this_subjectivity_abs = np.abs(val_li).sum(0)
    except:  # no_hashtag_text为空
        this_sentiment, this_subjectivity, this_sentiment_abs, this_subjectivity_abs = [
            None] * 4

    this_positive = positive(no_hashtag_text, threshold=0.1)
    # this_mood = mood(no_hashtag_text)
    # this_modality = modality(no_hashtag_text)

    # Text_NoHashtag_Sentiment_assessments.append(this_assessments)
    Text_NoHashtag_Sentiment_sentiment.append(this_sentiment)
    Text_NoHashtag_Sentiment_subjectivity.append(this_subjectivity)
    Text_NoHashtag_Sentiment_sentiment_abs.append(this_sentiment_abs)
    Text_NoHashtag_Sentiment_subjectivity_abs.append(this_subjectivity_abs)
    Text_NoHashtag_Sentiment_positive.append(this_positive)
    # Text_NoHashtag_Sentiment_mood.append(this_mood)
    # Text_NoHashtag_Sentiment_modality.append(this_modality)

    # spacy 名词动词提取
    doc = nlp(no_hashtag_text)
    noun_tokens = [
        token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    verb_tokens = [token.text for token in doc if token.pos_ == 'VERB']

    Text_NounPhrases.append(';'.join(noun_tokens))
    Text_Verbs.append(';'.join(verb_tokens))


df2['Text_Emoji'] = Text_Emoji
df2['Text_Emoji_Number'] = Text_Emoji_Number

df2['Text_Hashtags'] = Text_Hashtags
df2['Text_NoHashtag'] = Text_NoHashtag
df2['Text_NoHashtag_Length'] = Text_NoHashtag_Length

# df2['Text_NoHashtag_Sentiment_assessments'] = Text_NoHashtag_Sentiment_assessments
df2['Text_NoHashtag_Sentiment_sentiment'] = Text_NoHashtag_Sentiment_sentiment
df2['Text_NoHashtag_Sentiment_sentiment_abs'] = Text_NoHashtag_Sentiment_sentiment_abs
df2['Text_NoHashtag_Sentiment_subjectivity'] = Text_NoHashtag_Sentiment_subjectivity
df2['Text_NoHashtag_Sentiment_subjectivity_abs'] = Text_NoHashtag_Sentiment_subjectivity_abs


df2['Text_NoHashtag_Sentiment_positive'] = Text_NoHashtag_Sentiment_positive
# df2['Text_NoHashtag_Sentiment_mood'] = Text_NoHashtag_Sentiment_mood
# df2['Text_NoHashtag_Sentiment_modality'] = Text_NoHashtag_Sentiment_modality
df2['Text_NounPhrases'] = Text_NounPhrases
df2['Text_Verbs'] = Text_Verbs


# =============== 1.2 图像处理 ===============

print('2.2 图像处理')
imgUrl_li = df2['Img_URL'].values
Img_labels = get_labelScore_by_api(imgUrl_li, maxResults=10)
df2['Img_labels'] = Img_labels


# # =============== 1.3 爬虫处理 ===============
# Owner_Account = []
# Owner_Username = []
# Owner_Followers = []
# Owner_Posts_Number = []
# Owner_Intro = []

# for idx, owner in tqdm(enumerate(df2.Owner, start=1), desc='2.3爬虫处理', total=len(df2)):
#     try:
#         account = instagram.get_account_by_id(owner)
#         Owner_Account.append(account.identifier)
#         Owner_Username.append(account.username)
#         Owner_Followers.append(account.followed_by_count)
#         Owner_Posts_Number.append(account.media_count)
#         Owner_Intro.append(account.biography)

#     except:
#         import traceback
#         traceback.print_exc()
#         import ipdb
#         ipdb.set_trace()
#         a = 1

# df2['Owner_Account'] = Owner_Account
# df2['Owner_Username'] = Owner_Username
# df2['Owner_Followers'] = Owner_Followers
# df2['Owner_Posts_Number'] = Owner_Posts_Number
# df2['Owner_Intro'] = Owner_Intro

# =============== 1.4 保存结果 ===============
df2.to_csv('第二阶段结果.csv', encoding='utf-8-sig', index=None)
