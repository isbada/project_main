'''
GOOGLE 识图API
'''


# json_data = {
#     "requests": [
#         # ------------------------- request1 -------------------------------
#         {
#             "image": {
#                 "source": {
#                     "imageUri":
#                     "https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/p1080x1080/77288138_556443745135565_1593771537751550604_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=105&oh=dacc44db11b7d59132391e0afdac2531&oe=5EAF958D"
#                 }
#             },
#             "features": [
#                 {
#                     "type": "LABEL_DETECTION",
#                     "maxResults": 10
#                 }
#             ]
#         },
#         # ------------------------- request2 --------------------------------
#         {
#             "image": {
#                 "source": {
#                     "imageUri":
#                     "https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/fr/e15/s1080x1080/76886184_293395618251407_7993895929839704897_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=111&oh=d3c3f811eb61902e4e97a3a247a31ab0&oe=5E7658E7"
#                 }
#             },
#             "features": [
#                 {
#                     "type": "LABEL_DETECTION",
#                     "maxResults": 10
#                 }
#             ]
#         }
#     ]
# }


import requests
from copy import deepcopy
from tqdm import tqdm
import numpy as np

API_URL = 'https://vision.googleapis.com/v1/images:annotate?key=AIzaSyB6T6Hx1RRxrNTW_6Nq2LAxaRhYyPo1xZM'

JSON = {
    "requests": []
}

REQUEST_SINGLE = {
    "image": {
        "source": {
            "imageUri": None
        }
    },
    "features": [
        {
            "type": "LABEL_DETECTION", "maxResults": None
        }
    ]
}


def get_labelScore_by_api(url_li, maxResults=10, num_per_req=16):
    '''
    传入url列表，返回标签和概率
    @num_per_req 每次请求的最大数量 最大值是16  超过16报错
    '''

    LABEL_SCORE_LI = []
    for idx in tqdm(np.arange(0, len(url_li), num_per_req), desc='GoogleAPI请求图片分类结果中...'):
        url_li_slice = url_li[idx:idx + num_per_req]

        json_data = deepcopy(JSON)
        for url in url_li_slice:  # 填入每一个参数
            req = deepcopy(REQUEST_SINGLE)
            req['image']['source']['imageUri'] = url
            req['features'][0]['maxResults'] = maxResults
            json_data['requests'].append(req)

        res = requests.post(API_URL, json=json_data).json()
        for ele in res['responses']:

            # 正常返回结果有 labelAnnotations 属性
            if 'labelAnnotations' in ele.keys():
                label_scores = [(dic['description'], dic['score'])
                                for dic in ele['labelAnnotations']]
            # 返回结果有错误信息，一般是链接失效
            elif 'error' in ele.keys():
                label_scores = ele['error']['message']
            # 分类结果就是个空的
            else:
                label_scores = None

            LABEL_SCORE_LI.append(label_scores)

    return LABEL_SCORE_LI


if __name__ == '__main__':
    url_li = ['https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/p1080x1080/77288138_556443745135565_1593771537751550604_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=105&oh=dacc44db11b7d59132391e0afdac2531&oe=5EAF958D',
              'https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/fr/e15/s1080x1080/76886184_293395618251407_7993895929839704897_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=111&oh=d3c3f811eb61902e4e97a3a247a31ab0&oe=5E7658E7',
              'https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/76902303_2433927036730034_2376869606126233931_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=102&oh=7f41b086d82b1f40c49998fbacfc3538&oe=5E726B4A',
              'https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/s1080x1080/74698413_833293840442794_9019736844709968265_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=110&oh=e9a972cd61573e928e7f4b4a0476ee21&oe=5E7F67F9',
              'https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/73141838_2183927311909521_3616582813794707003_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=109&oh=e1e333ad3ccc2eb577a34012fb01843c&oe=5EB23557',
              'https://scontent-amt2-1.cdninstagram.com/v/t51.2885-15/e35/75562958_162056521680274_2416220052354053564_n.jpg?_nc_ht=scontent-amt2-1.cdninstagram.com&_nc_cat=110&oh=a5e2e7c334b2f7697c54c0bbe3865377&oe=5EB45F94']

    get_labelScore_by_api(url_li, maxResults=10)
