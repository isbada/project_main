# 1.豆瓣影评-白蛇缘起

- 任务描述: 

     目标url: `https://movie.douban.com/subject/30331149/comments`


- 实现细节:
    1. 分析请求数据，采取header加cookie方法, 伪装请求为浏览器头
    
    2. 简单词频统计，使用`pyechart`接口输出统计结果如下所示
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjhhp0ewxj30wi0io432.jpg)
    
    3. 词云图绘制，添加蒙板形状，使用自定义字体
    ```{python}
    wc = WordCloud(
        background_color='#FFF0F5',  # 设置背景颜色
        mask=mask,  # 设置背景图片
        font_path='./MacFSGB2312.ttf',  # 若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
        random_state=30  # 设置有多少种随机生成状态，即有多少种配色方案
    )
    ```
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjhfys004j30ta0rqnbk.jpg)

- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjhgpi14rj30rv0h148a.jpg)



# 2.豆瓣影评-白蛇缘起


- 任务描述: 

     - 目标url: `https://book.douban.com/top250`
     - top10书籍词频分析，可视化


- 实现细节:
    1. step1 获取top10书籍的索引信息，step2 获取每本书籍的短评、打分信息
    
    2. BeautifulSoup+正则解析爬取到的信息
    
    3. 基于pandas绘图接口的可视化
    
    4. pyecharts和pandas绘图接口的使用

- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjhq2kmedj314u0l6400.jpg)
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjhqp0xj9j30xj0eqtfd.jpg)

