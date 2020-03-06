# Instgram数据整理

- 任务描述: 
    - 详细任务见文档`需求文档csv.pdf`
    - 设计 NLP CV等多项操作，主要是接口的调用
    - 原始数据sample
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckgtqj1aoj31ge0u0asd.jpg)


- 实现细节:
    
    1. 第一阶段
        - “Text”中是否含有网址/电话/stopwords 正则（注意网址正则方式）
        - 重复: df.Series.duplicated
        - 主导rgb值: `get_dominant_color`函数
    
    2. 第二阶段
        - Emoji提取`re.compile('[\U00010000-\U0010ffff]')`
        - Hashtag提取`re.compile('#\w+')`
        - Sentiment分析`pattern.sentimen`(pattern包)
        - 名词动词提取
            ```{python}
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(no_hashtag_text)
            noun_tokens = [
                token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            verb_tokens = [token.text for token in doc if token.pos_ == 'VERB']
            ```
        - 图像识别：GoogleAPI `https://vision.googleapis.com/`
            - 传入url或者本地图像均可
            - 每次最大请求数量：16
            - 返回参数包含API对图像进行分类的文本标签，可定义最大返回数量`maxResults`

