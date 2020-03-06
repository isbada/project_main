# 某学校官网新闻浏览器

- 功能描述: 

     1. 读本地数据，列表显示（Reportsinfo.xlsx）
     2. 爬取网络实时数据，更新1的列表(从`http://news.bnu.edu.cn/zx/ttgz/index.htm`网页中通过爬虫把上述数据爬下来，并保存为Excel文件)
     3. 根据浏览次数的排序，对其中浏览次数最高的报道，可以获取报道文字和图片内容，以及词云信息
     4. 简单统计: 每年的头条关注报道的总数量、以及每一年中每个月的头条关注报道的数量


- 实现细节:
    1. requests数据获取 + tkinter界面设计
    
    2. GUI界面的进度显示：`tqdm_gui`
    ```{python}
    for page in tqdm_gui(range(int(page_num)), desc='Spidering Data', ascii=False):
        # for page in tqdm_gui(range(int(1)), desc='爬取头条关注数据', ascii=False):
        page = '' if page == 0 else str(page)
        URL = URL_FMT.format(page=page)
        res = requests.get(URL)
        ...
    ```
    
    3. tkinter的图像操作细节
    ```{python}
    # 图像初始化
    self.someimg = ImageTk.PhotoImage(Image.open('res/1.jpg')

    # 图像更新（两个步骤缺一不可）
    self.label_img.config(image=img)
    self.label_img.image = img
    ```
    

- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcji9gnluzj315g0u0npe.jpg)
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjijmuezyj30yu0ig176.jpg)
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjik0x1pdj31js0u0tb7.jpg)

