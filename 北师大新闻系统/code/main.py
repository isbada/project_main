import tkinter as tk
import tkinter.messagebox as tkMessagebox
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
from tqdm import tqdm, tqdm_notebook, tqdm_gui
from PIL import Image, ImageTk
from io import BytesIO
import jieba
from wordcloud import WordCloud
import os
import shutil
# from skimage import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


# -------------------------------- 全局变量 --------------------------------

p_view_num = re.compile("write\('(\d+)'\)")

IMG_SHAPE = (1000, 500)  # 大图尺寸

OUT_PATH = './浏览次数最高的报道/'  # 浏览次数最高的报道输出路径


# -------------------------------- 辅助函数 --------------------------------


def spider_index():
    '''函数 -- 在线爬取新闻索引信息的pandas.DataFrame'''
    URL_FMT = 'http://news.bnu.edu.cn/zx/ttgz/index{page}.htm'

    # 获取页数
    res = requests.get(URL_FMT.format(page=''))
    soup = BeautifulSoup(res.content.decode('utf8'), 'html.parser')
    page_num = soup.find('div', {'class': 'pages'}).span.text.split('/')[1]

    DATES, TITLES, LINKS, VIEW_NUMBERS = [], [], [], []

    # for page in tqdm(range(int(page_num)), desc='爬取头条关注数据'):
    for page in tqdm_gui(range(int(page_num)), desc='Spidering Data', ascii=False):
        # for page in tqdm_gui(range(int(1)), desc='爬取头条关注数据', ascii=False):
        page = '' if page == 0 else str(page)
        URL = URL_FMT.format(page=page)
        res = requests.get(URL)
        soup = BeautifulSoup(res.content.decode('utf8'), 'html.parser')
        # 获取日期、标题、链接、浏览次数
        articles = soup.find_all('li', {'class': 'item-info01'})
        dates = [ele.find('span', {'class': 'time'}).text for ele in articles]
        titles = [ele.find('h3').text for ele in articles]
        links = [urljoin(URL, ele.a.get('href')) for ele in articles]
        views_links = [urljoin(URL, ele.find('script').get('src'))
                       for ele in articles]
        view_numbers = [p_view_num.search(requests.get(
            link).text).group(1) for link in views_links]

        DATES.extend(dates)
        TITLES.extend(titles)
        LINKS.extend(links)
        VIEW_NUMBERS.extend(view_numbers)

    df_spider = pd.DataFrame(
        {'日期': DATES, '标题': TITLES, '链接': LINKS, '浏览次数': VIEW_NUMBERS})

    # 存储数据
    df_spider.to_csv('res/spider_Reportsinfo.csv',
                     encoding='utf-8-sig', index=None)


def spider_content(url):
    '''函数 -- 爬取新闻详情页信息'''

    res = requests.get(url)
    soup = BeautifulSoup(res.content.decode('utf8'), 'html.parser')

    title = soup.find('div', {'class': 'articleTitle'}).text
    author = soup.find('div', {'class': 'articleAuthor'}).text
    content = soup.find('div', {'class': 'article'}).text
    image_soups = soup.find('div', {'class': 'article'}).find_all('img')
    image_links = [urljoin(url, img.get('src')) for img in image_soups]

    return title, author, content, image_links


# -------------------------------- 界面设计 --------------------------------


class App(tk.Frame):
    """docstring for App"""

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.df_local = pd.read_excel('res/Reportsinfo.xlsx')
        self.df = self.df_local  # 记录当前操作的数据
        self._init_var()
        self._create_widgets()

    def _init_var(self):
        '''初始化变量
        @self.algorithm 选择的算法
        '''

        # 初始化一些值
        self.img_idx = 0  # 图片索引
        self.RAW_IMG_LI = [ImageTk.PhotoImage(Image.open('res/1.jpg').resize(IMG_SHAPE)),
                           ImageTk.PhotoImage(Image.open(
                               'res/2.jpg').resize(IMG_SHAPE)),
                           ImageTk.PhotoImage(Image.open('res/3.jpg').resize(IMG_SHAPE))]
        self.img_li = self.RAW_IMG_LI
        self.lb_idx = 0  # listbox索引

    def prev_img(self):
        '''显示上一张图'''
        N = len(self.img_li)
        self.img_idx = (self.img_idx - 1) % N
        img = self.img_li[self.img_idx]
        self.label_img.config(image=img)
        self.label_img.image = img

    def next_img(self):
        '''显示下一张图'''
        N = len(self.img_li)
        self.img_idx = (self.img_idx + 1) % N
        img = self.img_li[self.img_idx]
        self.label_img.config(image=img)
        self.label_img.image = img

    def insert2listbox(self, li):
        '''函数-将li内容插入listbox'''
        self.lb.delete(0, tk.END)
        for i in range(len(li)):
            self.lb.insert(tk.END, str(li[i]))

    def fun1(self):
        '''本地获取报道列表'''
        self.df = self.df_local
        li = ['{} {} {} {}'.format(*row) for row in self.df.values]
        self.insert2listbox(li)

    def fun2(self):
        '''根据浏览次数排序'''
        df = self.df.copy()
        df = df.sort_values(by='浏览次数', ascending=False)
        li = ['{} {} {} {}'.format(*row) for row in df.values]
        self.insert2listbox(li)
        self.lb.select_set(0)  # 排序后重新设置索引为0

        # 不存在[浏览次数最高的报道]则新建文件夹,确保路径存在
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
        else:  # 存在则清空
            shutil.rmtree(OUT_PATH)
            os.makedirs(OUT_PATH)

        # 根据浏览次数对头条关注报道进行逆序排序，对其中浏览次数最高的报道，爬取报道文字内容保存为一个txt文件，爬取其中的图片，保存在电脑里
        url = df.iloc[0]['链接']
        title, author, content, image_links = spider_content(url)
        img_li = [requests.get(link).content for link in image_links]
        img_names = [link.split('/')[-1] for link in image_links]

        with open(OUT_PATH + '文字内容.txt', 'w') as fout:
            fout.write(title + author + content)
        for name, img in zip(img_names, img_li):
            with open(OUT_PATH + name, 'wb') as file:
                file.write(img)
        # ta
        mytext = " ".join(jieba.cut(content))
        wordcloud = WordCloud(
            scale=5, font_path="res/KaiTi.ttf").generate(mytext)
        wordcloud.to_file(OUT_PATH + "词云图.png")

    def fun3(self):
        '''每年的报道量统计'''
        df = self.df.copy()
        years = list(map(lambda tstr: pd.datetime.strptime(
            tstr, '%Y-%m-%d').year, df['日期']))
        df['年份'] = years
        df_plot = df.groupby('年份')['标题'].count()
        # import ipdb
        # ipdb.set_trace()
        df_plot.plot(kind='bar', figsize=(10, 6), rot=0)
        plt.ylabel('报道量')
        plt.title('每年的报道总量统计')
        plt.show()

    def fun4(self):
        '''逐年每月的报道量统计'''
        df = self.df.copy()
        years = list(map(lambda tstr: pd.datetime.strptime(
            tstr, '%Y-%m-%d').year, df['日期']))
        months = list(map(lambda tstr: pd.datetime.strptime(
            tstr, '%Y-%m-%d').month, df['日期']))
        df['年份'] = years
        df['月份'] = months

        N = len(set(years))  # 年数
        u_years = sorted(set(years))
        fig, axes = plt.subplots(1, N, figsize=(12, 6))
        for i in range(N):
            this_ax = axes[i]
            this_year = u_years[i]
            df_tmp = df[df['年份'] == this_year].groupby(['月份'])['标题'].count()
            df_tmp.plot(kind='bar', ax=this_ax, title=f'{this_year}年每个月的报道量')
            this_ax.set_ylabel('报道量')

        plt.show()

    def fun5(self):
        '''网上获取报道列表'''
        spider_index()
        self.df_spider = pd.read_csv('./res/spider_Reportsinfo.csv')
        self.df = self.df_spider
        li = ['{} {} {} {}'.format(*row) for row in self.df.values]
        self.insert2listbox(li)

    def fun6(self):
        '''查看报告内容'''

        idx = self.lb.curselection()[0]
        url = self.lb.get(idx).split()[2]
        title, author, content, _ = spider_content(url)

        # 清除text后插入新的text
        self.text.delete(1.0, tk.END)
        self.text.insert(1.0, title + author + content)

    def fun7(self):
        '''查看报告词云'''

        idx = self.lb.curselection()[0]
        url = self.lb.get(idx).split()[2]
        title, author, content, _ = spider_content(url)

        mytext = " ".join(jieba.cut(content))
        wordcloud = WordCloud(
            scale=5, font_path="res/KaiTi.ttf").generate(mytext)
        plt.figure('词云图', figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def fun8(self):
        '''查看报告中图片'''
        idx = self.lb.curselection()[0]
        url = self.lb.get(idx).split()[2]
        title, author, content, image_links = spider_content(url)

        if image_links:
            # img_li = [io.imread(link) for link in image_links]
            img_li = [requests.get(link).content for link in image_links]
            img_li = [Image.open(BytesIO(img)).resize(IMG_SHAPE)
                      for img in img_li]
            NEW_IMG_LI = [ImageTk.PhotoImage(img) for img in img_li]
            self.img_li = NEW_IMG_LI
        else:
            tkMessagebox.showerror(title='错误', message='选中的文章没有图片')
            self.img_li = self.RAW_IMG_LI

        self.next_img()

    def _create_widgets(self):
        '''
        创建初始化视图的函数
        '''

        # frame1第1行，放列表和正文
        frame1 = tk.Frame(self)
        self.lb = tk.Listbox(frame1, height=12, width=5, font=('', 8))
        self.fun1()
        self.lb.select_set(0)
        self.text = tk.Text(frame1, height=12, width=5, font=('', 8))
        # --------------------------------
        frame1.pack(side=tk.TOP, fill=tk.BOTH)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # frame2第2行
        frame2 = tk.Frame(self)
        btn_fun1 = tk.Button(
            frame2, width=5, text='本地获取报道列表', command=self.fun1)
        btn_fun2 = tk.Button(
            frame2, width=5, text='根据浏览次数排序', command=self.fun2)
        btn_fun3 = tk.Button(
            frame2, width=5, text='每年的报道量统计', command=self.fun3)
        btn_fun4 = tk.Button(
            frame2, width=5, text='逐年每月的报道量统计', command=self.fun4)
        # --------------------------------
        frame2.pack(side=tk.TOP, fill=tk.X)
        for btn in [btn_fun1, btn_fun2, btn_fun3, btn_fun4]:
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # frame3第3行
        frame3 = tk.Frame(self)
        btn_fun5 = tk.Button(
            frame3, width=5, text='网上获取报道列表', command=self.fun5)
        btn_fun6 = tk.Button(frame3, width=5, text='查看报告内容', command=self.fun6)
        btn_fun7 = tk.Button(frame3, width=5, text='查看报告词云', command=self.fun7)
        btn_fun8 = tk.Button(
            frame3, width=5, text='查看报告中图片', command=self.fun8)
        # --------------------------------
        frame3.pack(side=tk.TOP, fill=tk.X)
        for btn in [btn_fun5, btn_fun6, btn_fun7, btn_fun8]:
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # frame4第4行，放三张显示图
        frame4 = tk.Frame(self)
        img = self.img_li[self.img_idx]
        self.label_img = tk.Label(frame4, image=img)
        # --------------------------------
        frame4.pack(side=tk.TOP)
        self.label_img.pack()

        # frame5第5行，放上一张下一张按键
        frame5 = tk.Frame(self)
        btn_pre_img = tk.Button(
            frame5, width=20, text='上一张图片', command=self.prev_img)
        btn_next_img = tk.Button(
            frame5, width=20, text='下一张图片', command=self.next_img)
        # --------------------------------
        frame5.pack(side=tk.TOP, fill=tk.X)
        btn_pre_img.pack(side=tk.LEFT)
        btn_next_img.pack(side=tk.RIGHT)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('北师大头条关注相关报道')
    # root.geometry('650x400')
    root.resizable(False, False)  # 不支持缩放
    App(root).pack()
    root.mainloop()
