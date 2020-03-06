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
from tqdm import tqdm

from sklearn.cluster import KMeans
from collections import Counter
import cv2  # for resizing image
import os


# ------------------------全局变量-------------------------------
# p_url = re.compile(
#     r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))')

p_url = re.compile(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""")

p_tel = re.compile('Tel\:\ ?\d+')

# nltk_stopwords={'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren','as','at','be','because','been','before','being','below','between','both','but','by','can','couldn','d','did','didn','do','does','doesn','doing','don','down','during','each','few','for','from','further','had','hadn','has','hasn','have','haven','having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn','it','its','itself','just','ll','m','ma','me','mightn','more','most','mustn','my','myself','needn','no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan','she','should','shouldn','so','some','such','t','than','that','the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn','we','were','weren','what','when','where','which','while','who','whom','why','will','with','won','wouldn','y','you','your','yours','yourself','yourselves'}

stopwords_ads = [line.strip().lower()
                 for line in open('stopwords_ads.txt')]  # 要小写化处理

p_img_name = re.compile('([^/]*?)\?')


# 清空url.txt / tel.txt / wrong_img.txt
with open('url.txt', 'w') as fout:
    fout.write('')

with open('tel.txt', 'w') as fout:
    fout.write('')

with open('wrong_img.txt', 'w') as fout:
    fout.write('')


# ------------------------辅助函数-------------------------------


def has_stopword(s):
    '''检测字符串是否有stopwords_ads里面的词条'''
    s = s.lower()
    s_li = re.split('\s+', s)
    for w in stopwords_ads:
        if len(w.split()) > 1:
            if w in s:
                return w
        else:
            if w in s_li:
                return w
    else:
        return False


def get_dominant_color(image, k=4, image_processing_size=None):
    """
    获取dominant color的函数
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def dominant_color_diff(tup):
    '''检测r g b 中最大值与最小值是否大于150'''
    v_max, v_min = max(tup), min(tup)
    if v_max - v_min > 150:
        return True
    else:
        return False

# ----------------------------第一个表格---------------------------------


# 读取原始表格
df = pd.read_csv('babyfeeding.csv', index_col=0)
df1 = df.copy()

# =============== 1.1 检测文本 ===============
Text_Include_Url = []
Text_Include_Tel = []
Text_Include_Stopwords = []
Text_Repetition = []


for idx, text in tqdm(enumerate(df.Text, start=1), desc='1.1文本处理', total=len(df)):
    # 1.1.1 URL检测
    match1 = p_url.search(text)
    if match1:
        text = match1.group()
        with open('url.txt', 'a') as fout:
            fout.write(text + '\n')
        Text_Include_Url.append(True)
    else:
        Text_Include_Url.append(False)

    # 1.1.2 Tel检测
    match2 = p_tel.search(text)
    if match2:
        text = match2.group()
        with open('tel.txt', 'a') as fout:
            fout.write(text + '\n')
        Text_Include_Tel.append(True)
    else:
        Text_Include_Tel.append(False)

    # 1.1.3 stopwords检测
    match3 = has_stopword(text)
    Text_Include_Stopwords.append(match3)


# 1.1.4 重复程度检测(检测重复出现的文本，不然计算量太大)
Text_Repetition = df.Text.duplicated().values

Text_Include_Url = np.array(Text_Include_Url)
Text_Include_Tel = np.array(Text_Include_Tel)
Text_Include_Stopwords = np.array(Text_Include_Stopwords)


df1['Text_Include_Url'] = Text_Include_Url
df1['Text_Include_Tel'] = Text_Include_Tel
df1['Text_Include_Stopwords'] = Text_Include_Stopwords
df1['Text_Repetition'] = Text_Repetition

print(f'URL检测筛选数据{sum(Text_Include_Url)}条')
print(f'Tel检测筛选数据{sum(Text_Include_Tel)}条')
# print(f'Stopwords检测筛选数据{sum(Text_Include_Stopwords!="False")}条')
print(f'Stopwords检测筛选数据{sum(Text_Include_Stopwords!="False")}条')
print(f'Repetition检测筛选数据{sum(Text_Repetition)}条')


# =============== 1.2 检测图像 ===============
Img_Dominant_Color_RGB = []
Img_Dominant_Color_Difference = []
for idx, img_url in tqdm(enumerate(df.Img_URL, start=1), desc='1.2图像处理', total=len(df)):
    try:
        img_name = p_img_name.search(img_url).group(1)
        img = cv2.imread(os.path.join('babyfeeding_img', img_name))
        color_RGB = get_dominant_color(
            img, k=4, image_processing_size=(25, 25))
        color_diff = dominant_color_diff(color_RGB)

        Img_Dominant_Color_RGB.append(color_RGB)
        Img_Dominant_Color_Difference.append(color_diff)

    except:
        # import traceback
        # traceback.print_exc()
        # import ipdb
        # ipdb.set_trace()
        # a = 1
        with open('wrong_img.txt', 'a') as fout:
            fout.write(img_url + '\n')


# Img_Dominant_Color_RGB = np.array(Img_Dominant_Color_RGB)
Img_Dominant_Color_Difference = np.array(Img_Dominant_Color_Difference)


df1['Img_Dominant_Color_RGB'] = Img_Dominant_Color_RGB
df1['Img_Dominant_Color_Difference'] = Img_Dominant_Color_Difference

print(f'Dominant_Color_Difference检测筛选数据{sum(Img_Dominant_Color_Difference)}条')


# =============== 1.3 保存表1 ===============
mask = Text_Include_Url | Text_Include_Tel | (
    Text_Include_Stopwords != 'False') | Text_Repetition | Img_Dominant_Color_Difference

print(f'共筛选数据{sum(mask)}条')


# df1 = df1.iloc[mask, :]
# 保存结果
df1.to_csv('第一阶段结果.csv', index=None)
