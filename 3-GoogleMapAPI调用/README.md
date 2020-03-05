# 谷歌API调用，获取经纬度，计算导航距离

- 任务描述: 

     `sites.txt`中的数据代表New York的地名，需要使用google Map进行模糊匹配得到两两地址之间的距离


- 方法分析:
    1. GoogleMap Gecoding服务提供了相关的接口，可以直接模糊匹配两点间的距离
    ```{python}
    # 地址获取经纬度
    API_FORMAT = 'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={key}'

    # 两个地址之间导航距离
    API_FORMAT = 'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&sensor=false&key={key}'
    ```

    2. 为了提高地名匹配精度，需要在地名后面加上new york，加强定位精准度
    
    3. 263个地址，总计查询量<a href="https://www.codecogs.com/eqnedit.php?latex=C_{263}^{2}=34453" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_{263}^{2}=34453" title="C_{263}^{2}=34453" /></a>（两两之间的匹配只用做一次），多线程加速后需要5min.

    4.注意控制线程数量，免费的Google API服务有访问速度限制。

- 结果展示
![](https://tva1.sinaimg.cn/large/00831rSTgy1gcjedoqkjaj30tb0gfjuy.jpg)



