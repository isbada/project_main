# 银保监数据爬虫

- 目标url: `http://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=931&itemUrl=zhengwuxinxi/xingzhengchufa.html&itemName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A`

- 任务描述: 
三级机关中，包含“信息公开表”的信息提取，收集2017-2019年数据
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjc7d0p0bj315y0u0hdt.jpg)

- 方法分析:
    1. 分析后台XHR请求，可以得到下面两个数据传递的URL
    ```{python}
    # 取索引（json） 数据URL
    URL1 = 'http://www.cbirc.gov.cn/cbircweb/DocInfo/SelectDocByItemIdAndChild'

    # 取详情（json） 数据URL
    URL2 = 'http://www.cbirc.gov.cn/cn/static/data/DocInfo/SelectByDocId/data_docId={docId}.json'

    ```
    
    2. 难点: 
        - 由于年份久远，有的表格格式不统一(尤其是分局机关)，需要编辑不同的正则表达式进行多次调试
        - web的表格格式含有很多标签，对策是用BeautifulSoup清洗标签，然后对剩下的字符串进行正则提取
        - 数据量在1w左右,单个URL请求速度太慢，采取了多线程爬虫，几分钟即可
        - 有罚款信息的，需要匹配提取信息中的金额数字，并解析为浮点数，由于中文表达博大精深，因此采取正则的方式并不能完美解决，需要人工调整。
    
    3. 其他
        - 有的链接存在坏链，或者是内容为空，对于解析失败的链接，输出到日志文件中，方便调试
        - 感谢管理员不封IP

- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjd2sqlt3j31zo0u0npe.jpg)



