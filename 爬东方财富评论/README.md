#  东方财富网评论信息爬虫

- 目标url: `http://guba.eastmoney.com/list,zssh000001,f_1.html`

- 任务描述: 股民评论信息的收集整理（时间范围`2019-06-01~2019-12-31`）
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjcs63n3qj30wy0futh7.jpg)


- 方法分析:
    1. URL构成比较简单，直接替换URL中PAGE的索引，直接request就可以
    
    2. 虽然不用伪装IP头，但是目标网站对于高强度的访问会封IP，要控制好多线程的量，同时进行随机暂停，控制访问速度，否则会封IP。主要是预算有限，不舍得消耗代理IP...
    
    3. 由于数据量比较庞大，大多数评论比较短，索引页信息就可以涵盖到，所以只爬取了索引页信息。


- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjdlydlqlj30o40gqn4n.jpg)

