# -*- coding:utf-8 -*-


import requests
import pandas as pd
import time
import random
import re
import jieba
from bs4 import BeautifulSoup
from collections import Counter
from pyecharts import Bar
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签


# step1 获取top10书籍的索引信息
print('\nstep1 正在爬取索引页信息......')

p_nameurl = re.compile('<a href="(.*?)" onclick=.*? title="(.*?)"')


res = requests.get('https://book.douban.com/top250')


url_bookname_l = p_nameurl.findall(res.text)[:10]  # 取前10个书籍的信息

commenturl_bookname_l = list(
    map(lambda tup: (tup[0] + 'comments/', tup[1]), url_bookname_l))

print('索引页信息爬取完毕!')

# step2 获取每本书籍的短评、打分信息
print('\nstep2 正在爬取top10书籍短评信息......')

PAGE = 5  # 每本书爬取5页  5*20=100条短评
all_record = []
for cmturl, bookname in commenturl_bookname_l:
    for page in range(1, PAGE + 1):
        print('正在爬取书籍【%s】,第%d页' % (bookname, page))
        this_url = cmturl + 'hot?p=' + str(page)  # 构造目标url页面
        this_res = requests.get(this_url, timeout=10)
        soup = BeautifulSoup(this_res.text, 'html.parser')

        infos = soup.select('li[class="comment-item"]')  # 选取评论区域
        for info in infos:
            p_name = re.compile(
                'class="avatar">.*?<.*?title="(.*?)"', re.S)  # 用户名
            p_time = re.compile('(\d{4}-\d{2}-\d{2})', re.S)  # 评分时间
            p_score = re.compile('allstar(\d+)')  # 用户打分
            p_comment = re.compile('class="short">(.*?)<', re.S)  # 用户评论

            name = p_name.search(str(info))
            time_cmt = p_time.search(str(info))
            score = p_score.search(str(info))
            comment = p_comment.search(str(info))

            name = name.group(1) if name else None
            time_cmt = time_cmt.group(1) if time_cmt else None
            score = score.group(1) if score else None
            comment = comment.group(1).strip() if comment else None
            if not comment:  # 评论信息为空串干脆设置为None
                comment = None

            this_record = [bookname, name, time_cmt, score, comment]
            all_record.append(this_record)

        time.sleep(random.random() * 1)  # 每次爬完一个页面，随机暂停一下，规避反爬

df_all = pd.DataFrame(all_record, columns=['书名', '评论者', '评论时间', '打分', '评论详情'])
df_all['打分'] = df_all['打分'].fillna(0).astype(int)  # 评分信息，数据类型转换

df_all.to_csv('./top10书籍书评爬取结果.csv',
              encoding='utf_8_sig', index=None)  # 保存爬虫结果到文件

# df_all = pd.read_csv('./top10书籍书评爬取结果.csv')
# df_all = df_all[df_all['书名'].notna()]  # 删掉空行

# step3 数据分析
print('\nstep3 正在对爬虫结果进行数据分析......')
print('============ Top10书籍平均评分情况 ==============')
score_ser = df_all.groupby('书名')['打分'].mean()
print(score_ser)
score_ser.plot(kind='barh', figsize=(10, 10),
               title='Top10书籍评分情况条形图', rot=30, color='Red')
plt.xlabel('平均评分')
plt.ylabel('书籍')
plt.show()


print('============ Top10书籍词频分析 ==============')
for bookname in df_all['书名'].unique():
    cmt_ser = df_all[df_all['书名'] == bookname]['评论详情']
    text_all = ' '.join(cmt_ser)
    w_counter = Counter(filter(lambda x: len(
        x) > 2, jieba.cut(text_all)))  # 过滤掉长度小于2的词语

    print('书籍【%s】评论Top10高频词汇 ==>' % bookname)
    df_w_counter = pd.DataFrame(
        w_counter.most_common(10), columns=['词语', '出现次数'])
    print(df_w_counter)

    bar = Bar('书籍【%s】评论Top10高频词汇 ==>' % bookname)  # 新建柱状图

    attr, v = zip(*sorted(w_counter.items(), key=lambda x: x[1], reverse=True))

    bar = Bar('书籍【%s】评论Top10高频词汇' % bookname)
    bar.add("词频", attr[:10], v[:10],
            is_label_show=True, bar_category_gap="30%")
    bar.render("书籍【%s】评论Top10高频词汇统计图.html" % bookname)  # 渲染到文件


input('分析完成，回车退出....')
