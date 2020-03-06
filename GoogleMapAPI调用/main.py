# -*- coding:utf-8 -*-


import requests
import base64
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool
from requests.adapters import HTTPAdapter

s = requests.Session()
# max_retries:设置重试次数
s.mount('https://', HTTPAdapter(max_retries=3))


SITE_FILE = './sites.txt'  # 记录位置名称的文件
OUT_FILE = './result.csv'  # 记录结果的文件
AREA_NAME = ', New York'  # 加在地址上，加强定位精准度,逗号是分隔符

KEY = base64.b64decode(b'QUl6YVN5REd5MnoxMGU0Mkgyb0gwNk5TWEpTbzRtdDRsbENpbUo0')

# 地址获取经纬度
# API_FORMAT = 'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={key}'

# 两个地址之间距离
API_FORMAT = 'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&sensor=false&key={key}'


LANGUAGE = 'Chinese'  # 设置返回信息的语言,默认是英文
if LANGUAGE == 'Chinese':
    API_FORMAT += '&language=zh-CN'


def getNavDistance(para_tuple):
    '''单步调用API，获取两点之间导航距离'''
    i, j = para_tuple
    origin, destination = name_l[i], name_l[j]

    # print("%s - %s spider..." % (origin, destination)) # 显示正在爬的地点
    para = {'origin': origin + AREA_NAME, 'destination': destination +
            AREA_NAME, 'key': KEY.decode('utf8')}

    api_url = API_FORMAT.format(**para)

    this_navDistance = ""
    try:
        result = s.get(api_url, timeout=8).json()
        this_navDistance = result[' '][0]['legs'][0]['distance']['value'] / 1000
        this_navDistance = str(round(this_navDistance, 2))
        # this_navDistance = str(this_navDistance)
    except:
        this_navDistance = "--"

    # 结果赋值 ： 28.9 mi 或者 --
    # print("\r%s - %s spider...%s" % (origin, destination, this_navDistance))
    return(i, j, this_navDistance)


def load_sites(file):
    site_l = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            site_l.append(line)
    return sorted(set(site_l))


# 全局变量
name_l = load_sites(SITE_FILE)  # 地点名称列表
site_num = len(name_l)  # 地点数量


def main():
    # 存储结果的dataframe
    df = pd.DataFrame(
        np.array([''] * (site_num**2)).reshape((site_num, site_num)), columns=name_l, index=name_l)

    idx_comb = []
    # 计算可能的组合，两两组合，只爬取最小的值
    for i in range(site_num - 1):
        for j in range(i + 1, site_num):
            idx_comb.append((i, j))

    # 单线程请求数据
    # for para in tqdm(idx_comb, total=len(idx_comb), desc='Data Get'):
    #     getNavDistance(para)

    # 多线程请求数据
    pool = Pool(20)
    for i, j, dis in tqdm(pool.imap_unordered(getNavDistance, idx_comb), total=len(idx_comb), desc='Location Getting'):
        df.iloc[j, i] = dis
    df.to_csv(OUT_FILE)
    print("文件写入完毕！")


if __name__ == '__main__':
    main()
