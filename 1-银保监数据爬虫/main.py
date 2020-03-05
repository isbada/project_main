'''
目标网站

http://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=931&itemUrl=zhengwuxinxi/xingzhengchufa.html&itemName=行政处罚

# 取索引（json） 数据URL
URL = 'http://www.cbirc.gov.cn/cbircweb/DocInfo/SelectDocByItemIdAndChild'


# 取详情（json） 数据URL
URL1 = 'http://www.cbirc.gov.cn/cn/static/data/DocInfo/SelectByDocId/data_docId=889152.json'

'''
import requests
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
import re
from multiprocessing import Pool

# ------------------------辅助函数 全局变量---------------------------


'''
itemId
    银保监会机关 4113 银保监局本级 4114  银保监分局本级4115

pageIndex
    银保监会机关 1-9 银保监局本级 24-372  银保监分局本级24-467
'''

HEADERS = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Connection': 'keep-alive',
    # 'Cookie': '_gscu_525435890=81329665sn6l4b15; _gscbrs_525435890=1; yfx_c_g_u_id_10006849=_ck20021018143014633505451642775; yfx_f_l_v_t_10006849=f_t_1581329670460__r_t_1581396570323__v_t_1581396570323__r_c_1; _gscs_525435890=t81396570cfmmxg14|pv:18',
    'Host': 'www.cbirc.gov.cn',
    # 'Referer': 'http://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=4113&itemUrl=ItemListRightList.html&itemName=%E9%93%B6%E4%BF%9D%E7%9B%91%E4%BC%9A%E6%9C%BA%E5%85%B3&itemsubPId=931&itemsubPName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',

}

PARAMS = {
    'itemId': None,
    'pageSize': 18,
    'pageIndex': None
}
# 取索引（json） 数据URL
URL1 = 'http://www.cbirc.gov.cn/cbircweb/DocInfo/SelectDocByItemIdAndChild'

# 取详情（json） 数据URL
URL2 = 'http://www.cbirc.gov.cn/cn/static/data/DocInfo/SelectByDocId/data_docId={docId}.json'


NAMES = ['银保监会机关', '银保监局本级', '银保监分局本级']
IDS = [4113, 4114, 4115]
RANGES = [range(1, 9 + 1), range(24, 372 + 1), range(25, 467 + 1)]
# RANGES = [range(1, 3 + 1), range(24, 24 + 3 + 1), range(24, 24 + 3 + 1)] #测试sample


def spider_idx(args):
    '''爬取索引页信息 '''

    ID, PAGEIDX = args

    PARAMS['itemId'], PARAMS['pageIndex'] = ID, PAGEIDX

    res = requests.get(URL1, headers=HEADERS, params=PARAMS)
    data = res.json()['data']['rows']

    data = [(d['docId'], d['publishDate'], d['docSubtitle'], ID)
            for d in data if '信息公开表' in d['docSubtitle']]
    # for d in data:
    #     print(d[2])

    return data


# ------------------------爬取索引信息---------------------------


DATA1 = {NAME: [] for NAME in NAMES}  # 存储三个类目下的索引数据
for NAME, ID, RANGE in zip(NAMES, IDS, RANGES):

    # 单线程操作
    # for PAGEIDX in tqdm(RANGE, desc=f'{NAME}-索引数据爬取中'):
    #     data = spider_idx((ID, PAGEIDX))
    #     DATA1[NAME].extend(data)

    # 多线程操作
    pool = Pool(20)  # 10倍线程爬虫
    N = len(RANGE)
    args_li = list(zip([ID] * N, RANGE))
    for data in tqdm(pool.imap(spider_idx, args_li), desc=f'{NAME}-索引数据爬取中', total=N):
        DATA1[NAME].extend(data)


# ------------------------爬取具体文档信息---------------------------


# 正则提取选项
p1 = re.compile('决定书?文号(.*?)\t', re.S)
p2_1 = re.compile('单位名称(.*?)\t')
p2_2 = re.compile('法定代表人.*?姓名(.*?)\t')

p2_31 = re.compile('个人姓名(.*?)\t')  # op1 个人姓名
p2_32 = re.compile('被处罚当事人姓名(.*?)\t')  # op1 个人姓名

p3 = re.compile('(案由\）?|主要违法违规事实)(.*?)\t')


p4 = re.compile('行政处罚依据(.*?)\t')
# p5 = re.compile('\t行政处罚决定(.*?)\t')
p5 = re.compile('\t行政处罚决定(?!书文号)(.*?)\t')

p6 = re.compile('处罚(决定)?的?机关名称(.*?)\t')
p7 = re.compile('处罚决定的日期(.*?)\t')

p_money1 = re.compile('(合计|共计)(.*?)万元')
p_money2 = re.compile('罚款(.*?)万元')


# s2 = '作出处罚机关名称河南银保监局\t'


def spider_content(args):
    '''根据DATA1的数据，爬取详情页数据'''
    docId, date, title, itemId = args

    # 单个数据字典模板
    FMT = {'行政处罚决定书文号': None, '单位名称': None, '法定代表人姓名': None, '个人姓名': None, '主要违法违规事实（案由）': None,
           '行政处罚依据': None, '行政处罚决定': None, '金额（万元）': None, '作出处罚决定的机关名称': None, '作出处罚决定的日期': None}

    # 添加内容页链接，方便回顾检查
    FMT['链接地址'] = f'http://www.cbirc.gov.cn/cn/view/pages/ItemDetail.html?docId={docId}&itemId={itemId}&generaltype=9'

    try:

        # 解析详情页json数据
        url = URL2.format(docId=docId)
        res = requests.get(url)
        html = res.json()['data']['docClob']
        soup = BeautifulSoup(html, 'lxml')

        infos = [re.sub('\s+', '', tr.text) for tr in soup.find(
            'table').find_all('tr') if tr.text]
        info_str = ('\t'.join(infos) + '\t').replace('\n', '')
        # import ipdb
        # ipdb.set_trace()

        # 解析数据
        if p1.search(info_str):
            FMT['行政处罚决定书文号'] = p1.search(info_str).group(1).strip()
        if p2_1.search(info_str):
            FMT['单位名称'] = p2_1.search(info_str).group(1).strip()
        if p2_2.search(info_str):
            FMT['法定代表人姓名'] = p2_2.search(info_str).group(1).strip()

        if p2_31.search(info_str):
            FMT['个人姓名'] = p2_31.search(info_str).group(1).strip()
        elif p2_32.search(info_str):
            FMT['个人姓名'] = p2_32.search(info_str).group(1).strip()

        if p3.search(info_str):
            FMT['主要违法违规事实（案由）'] = p3.search(info_str).group(
                2).strip().replace('（案由）', '')
        if p4.search(info_str):
            FMT['行政处罚依据'] = p4.search(info_str).group(1).strip()
        if p5.search(info_str):
            tmp = p5.search(info_str).group(1).strip()
            FMT['行政处罚决定'] = tmp

            # 提取金额
            money = None
            if p_money1.search(tmp):
                money = p_money1.search(
                    tmp).group(2).replace('人民币', '').replace('罚款', '')
            elif p_money2.search(tmp):
                money = p_money2.search(
                    tmp).group(1).replace('人民币', '').replace('罚款', '')
            if money and len(money) > 15:  # 长度太长，肯定提取的不对
                money = None
            FMT['金额（万元）'] = money

        if p6.search(info_str):
            FMT['作出处罚决定的机关名称'] = p6.search(info_str).group(2).strip()
        if p7.search(info_str):
            dt_str = p7.search(info_str).group(1).strip()
            # dt = pd.datetime.strptime(dt_str, '%Y年%m月%d日')
            FMT['作出处罚决定的日期'] = dt_str

        return FMT

    except Exception as e:
        with open('tmp.txt', 'a') as fout:
            fout.write(FMT['链接地址'] + '\t' + e.args[0] + '\n')

        return None


# args = (856187, 'date', 'title', 4114)
# spider_content(args)
# quit()


for NAME in DATA1:
    CONTENT = []

    # 单线程操作
    # for args in tqdm(DATA1[NAME], desc=f'{NAME}-详情数据爬取中'):
    #     FMT = spider_content(args)
    #     CONTENT.append(FMT.copy())

    # 多线程操作
    pool = Pool(20)  # 10倍线程爬虫
    for fmt in tqdm(pool.imap(spider_content, DATA1[NAME]), desc=f'{NAME}-详情数据爬取中', total=len(DATA1[NAME])):
        if fmt:
            CONTENT.append(fmt.copy())

    df = pd.DataFrame(CONTENT)
    # df = df.sort_values(by='作出处罚决定的日期')
    df.to_csv(f'{NAME}.csv', encoding='utf-8-sig', index=None)
