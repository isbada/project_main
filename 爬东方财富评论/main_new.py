'''
http://guba.eastmoney.com/list,zssh000001_804.html

爬取东方财富股吧网上证指数吧2019/6/1到2019/12/31的评论贴内容和发帖时间

'''

import requests
from multiprocessing import Pool
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from urllib.parse import urljoin
import time
import random
# ----------------全局 辅助函数-------------------

# IDX_URL = 'http://guba.eastmoney.com/list,zssh000001_{PAGE}.html'

IDX_URL = 'http://guba.eastmoney.com/list,zssh000001,f_{PAGE}.html'

START = 1044  # 12-31起始页，随时间改动
OFFSET = 6515  # 一共这么多页 包含2019-06-01~2019-12-31时间段的信息
# OFFSET = 100


def get_url_title_time(div_soup):
    '''获取url和time,过滤不合格的数据'''
    url = div_soup.find('span', {'class': 'l3'}).a.get('href')

    url = urljoin(IDX_URL, url)
    dt = div_soup.find('span', {'class': 'l5'}).text
    title = div_soup.find('span', {'class': 'l3'}).a.get('title')

    date, time = dt.split()

    # 判断时间范围，合适返回
    if '06-01' <= date <= '12-31':
        return (url, title, '2019-' + dt)
    else:
        return None


def parse_idx_url(idx):
    '''处理索引页
    http://guba.eastmoney.com/list,zssh000001,f_800.html
    '''
    try:
        res = requests.get(IDX_URL.format(PAGE=idx))

        # 出现【沪深300ETF期权12月23日正式上市】代表被封IP了
        assert '沪深300ETF期权12月23日正式上市' not in res.text

        soup = BeautifulSoup(res.text, 'lxml')
        divs = soup.find_all('div', {'class': 'articleh normal_post'})
        url_times = [get_url_title_time(d) for d in divs]

        url_times = [tup for tup in url_times if tup]  # 排除None项目
        time.sleep(random.random() * 2)  # 暂停随机时间  防止封IP

        return url_times
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'{idx}页爬取失败')
        return None


# ----------------流程-------------------


pool = Pool(10)  # 10倍线程，多次测试结果  不要更改
URL_TITLE_DATETIME = set()

# 多线程处理
for utd in tqdm(pool.imap_unordered(parse_idx_url, range(START, START + OFFSET)), desc='索引信息收集中', total=OFFSET):
    if utd:  # utd不为空 更新到数据中
        URL_TITLE_DATETIME.update(utd)


# 构造存储结果的DateFrame
print('存储结果中...')
df = pd.DataFrame(URL_TITLE_DATETIME, columns=['链接', '标题', '时间'])
df = df.sort_values(by='时间')  # 按照时间排序
df.to_csv('最终结果.csv', encoding='utf-8-sig', index=None)  # 保存结果
