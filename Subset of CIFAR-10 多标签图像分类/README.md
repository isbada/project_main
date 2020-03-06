# Subset of CIFAR-10 多标签图像分类

- 任务描述: 
    - 多标签图像分类任务
    - 数据集（Subset of CIFAR-10）
        
        链接:https://pan.baidu.com/s/1JBY_shal2ErYl3r6G6jIEw  密码:gsal
        
        - trnImage, 32x32x3x10000 matrix, training images (RGB image)
        - trnLabel, 10000x1 matrix, training labels (1-10)
        - tstImage, 32x32x3x1000 matrix, testing images (RGB image)
        - tstLabel, 1000x1 matrix, testing labels (1-10)
    
    ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg2dzir1j30k20eawr5.jpg)


- 实现细节:
    1. 标签分布（分布均衡）
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg5xldcrj316k0ik76i.jpg)

    2. 图像特征提取（HOG特征）
        ```{python}
        def computeFeatures(image):
            # This function computes the HOG features with the parsed hyperparameters and returns the features as hog_feature. 
            # By setting visualize=True we obtain an image, hog_as_image, which can be plotted for insight into extracted HOG features.
            hog_feature, hog_as_image = skimage.feature.hog(image, visualize=True, block_norm='L2-Hys')
            return hog_feature, hog_as_image
        ```
        
    
    3. 对比模型和效果
        - SVM + HOG feature
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg782u9oj30re0s2dmd.jpg)
        - Kmeans + HOG feature
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg839k1qj30pi0rmjxt.jpg)
        - Gaussian Mixture Model + HOG feature
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg8xszfxj30po0s644r.jpg)
        - Neural Networks(sklearn MLP) + HOG feature
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg9fzhyjj30pk0twtfh.jpg)
        - Neural Networks + Transfer Learning
        ![](https://tva1.sinaimg.cn/large/00831rSTgy1gckg9w1etvj30ri0u045l.jpg)

    4. 其他
        实验环境 google lab（GPU）


- 结果展示
    评价标准: F1-score

    Model    | HOG feature | standarlize HOG feature | self-learning feature
    ---|---|---|---
    SVM | 0.41   | 0.60| -
    KMeans | 0.05| 0.06| -
    Gaussian Mixture Model |0.07|0.08 | -
    Neural Networks | 0.56| 0.48 | -
    Neural Networks + Transfer Learning | -  |- |0.65

- 回顾分析
    - Neural Networks + Transfer Learning的效果最好，能够达到0.65，但是考虑到神经网络需要较高的训练成本，所以使用 SVM+standarlize HOG feature(f1=0.6) 作为替代项也是一个不错的选择。
    - 1.使用standarlize对HOG feature进行处理后，在SVM模型上的效果有了较大提升，可以考虑使用要求中提到的Principal Component Analysis/Linear Discriminative Analysis或者其他方法多尝试一下，另外可对SVM模型超参数进一步优化，可能能够取得更高的效果；2.Neural Networks + Transfer Learning方法中，选用的ResNet18模型是在ImageNet图像集上进行预训练的，两个数据集之间的样本存在一定差异，可以尝试一下更多预训练模型，进一步发挥 Transfer Learning的优势，提升模型性能。





