#!/usr/bin/env python
# coding: utf-8


# In[1]:


import requests
import pandas as pd
import time
import random
from lxml import etree
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

# # 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 100  # 图片分辨率


# ## 全局变量设置

# In[3]:


HEADER = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Connection': 'keep-alive',
    'Cookie': 'gr_user_id=6d02a86f-63ab-473e-9b99-77236f956044; _vwo_uuid_v2=8FBC28D83931A87B8803E14EB4A6EF8D|86ee1c05e433a0e7334aadd78dd4c5f0; douban-fav-remind=1; __gads=ID=9b5cbbafc311d74a:T=1557974920:S=ALNI_MZ1MEiyzMlFMcW6w2Ha9uif08xouw; __utmc=30149280; __utmv=30149280.19782; ll="118254"; bid=CI-gsNW97zo; viewed="27077140_30283996_24703171_25779298_30231493_4907691_10590856_2101004_6709783_2058536"; OUTFOX_SEARCH_USER_ID_NCOO=1773020828.2606747; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1583422085%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DCAd7kuuRGT2t3Jb4650a9fdmrwmp-w4Ud0HCkv8vK0m5VgOF194Ey0TBArcOp0uc%26wd%3D%26eqid%3Dda0cdacd0001ce8a000000065e611a7e%22%5D; _pk_ses.100001.4cf6=*; __utma=30149280.783084243.1540472438.1578712286.1583422085.83; __utmz=30149280.1583422085.83.58.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.597470368.1583422085.1583422085.1583422085.1; __utmc=223695111; __utmz=223695111.1583422085.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmt=1; __yadk_uid=rzBh7HfAlxl9YdejctwRAMcKay421uv5; __utmt=1; __utmb=30149280.2.10.1583422085; _pk_id.100001.4cf6=ac81bf03032978f3.1583422085.1.1583422156.1583422085.; __utmb=223695111.8.10.1583422085',
    'Host': 'movie.douban.com',
    'Referer': 'https://movie.douban.com/subject/27052168/comments?start=0&limit=20&sort=new_score&status=P',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}


URL_BASE = 'https://movie.douban.com/subject/30331149/comments'

PARAMS = {
    'start': None,
    'limit': 20,
    'sort': 'new_score',
    'status': 'P',
    'comments_only': 1
}


NUM = 500  # 爬取条数
PAGE = 9999  # 需要爬取的页数,设置一个比较大的数字

p_comment = re.compile('<span class="short">(.*?)</span>')

p_username_link = re.compile(
    '<a href="(https://www.douban.com/people/.*?/)" class="">(.*?)</a>')
p_rating = re.compile(
    '<span class="allstar(\d+) rating" title=')
p_comment_time = re.compile('<span class="comment-time " title="(.*?)">')
p_agree = re.compile('<span class="votes">(.*?)</span>')
p_short = re.compile('<span class="short">(.*?)</span>', re.S)


# ## 爬虫主流程

# In[4]:


s = requests.Session()  # 创建会话，可以保持cookie
s.headers.update(HEADER)  # 默认headers参数


all_info = []
for page in range(PAGE):
    print('\r正在爬取第%d页,收集到评论%d条...' % (page + 1, len(all_info)), end='')
    try:
        PARAMS['start'] = page * 20
        this_para = PARAMS
        res = s.get(URL_BASE, headers=HEADER, params=this_para, timeout=20)
        import ipdb
        ipdb.set_trace()
        html = res.json()['html']

        if '还没有人写过短评' in html:  # 没有更多的页面了
            break

        # xpath方式解析页面
        root = etree.HTML(html)
        comments = root.xpath('//div[@class="comment-item"]')
        for comment_ele in comments:
            try:
                comment_html = etree.tounicode(comment_ele)
                # 评分信息
                ratings = p_rating.search(comment_html).group(1)
                # 姓名和姓名链接
                links, usernames = p_username_link.search(
                    comment_html).groups()
                # 评分时间
                comment_times = p_comment_time.search(comment_html).group(1)
                # 赞同数量
                agrees = p_agree.search(comment_html).group(1)
                # 评论详情
                shorts = p_short.search(comment_html).group(1)
                # -----本条信息-----
                this_info = [links, usernames, ratings,
                             comment_times, agrees, shorts]
                all_info.append(this_info)
            except:
                continue  # 出错了代表信息不完成，丢弃本条信息

        if len(all_info) >= NUM:  # 爬到500条，不爬了
            break
        time.sleep(random.random() * 3)  # 随机暂停一段时间，避免反爬

    except:
        import traceback
        traceback.print_exc()
        import ipdb
        ipdb.set_trace()

print('\n共收集到评论%d条...爬取完毕！' % (len(all_info)))


# ## 保存数据

# In[5]:


COLUMNS = ['用户名', '用户主页', '评分', '评分时间', '赞同数量', '评论详情']
df = pd.DataFrame(all_info, columns=COLUMNS)
# 存储数据到csv
df.to_csv('./爬取结果.csv', encoding='utf_8_sig')


# ## 分析数据

# 1.词频分析 国漫/经典/画面 和高频词汇

# In[11]:


text_all = ''
for comment_text in df['评论详情']:
    text_all += (comment_text + ' ')

words = ['国漫', '经典', '画面']
words_cnt = []
for w in words:
    words_cnt.append(text_all.count(w))

FMT = '{:^10}|{:^10}'
print(FMT.format('词语', '词频'))
for i in range(len(words)):
    print(FMT.format(words[i], words_cnt[i]))


# In[18]:


from collections import Counter
filtered_words = filter(lambda x: len(x) > 1, jieba.cut(text_all))
wordsCnt = Counter(filtered_words)
print('排名前二十的高频词汇统计表：')
df_wordsCnt = pd.DataFrame(wordsCnt.most_common(20), columns=['词语', '出现次数'])
df_wordsCnt


# In[27]:


from pyecharts import Pie
attr = df_wordsCnt['词语']
v1 = df_wordsCnt['出现次数']
pie = Pie("排名前二十的高频词汇统计图", title_pos="center")
pie.add("服装销量", attr, v1,
        is_label_show=True,
        center=[50, 50],  # 中心点位置
        legend_orient="vertical",
        legend_pos="left"
        )
# pie.render("pie.html")#渲染到文件
pie


# 2. 词云

# In[43]:


import PIL
mask = np.array(PIL.Image.open('mask.png'))
# backgroud_Image = imread('backgroud_Image.png')


wc = WordCloud(
    background_color='#FFF0F5',  # 设置背景颜色
    mask=mask,  # 设置背景图片
    font_path='./MacFSGB2312.ttf',  # 若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
    random_state=30  # 设置有多少种随机生成状态，即有多少种配色方案
)

wordcloud = wc.generate(text_all)
wordcloud.to_image()
