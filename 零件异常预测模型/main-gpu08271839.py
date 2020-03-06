import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 不显示警告
import warnings
warnings.filterwarnings("ignore")


# -------------------------- 全局变量 ------------------------------

# TQDM显示选项
TQDM = tqdm

# 训练/测试 数据存储路径
TRAIN_DIR, TEST_DIR = './train', './test'

# 故障标签长度(s)
TROUBLE_LEN = 0.5

# WINDOW时间长度(s)
WINDOW_LEN = 59

# 选择使用的特征列表
FEAT_SELECT = ['[8:2]', '[6:17]', '[1:7]', '[1:5]',
               '[1:20]', '[1:0]', '[1:1]', '[8:4]', '[1:133]']

SCALE = StandardScaler()

# 指定GPU/CPU参数
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print('DEVICE:', DEVICE)


# -------------------------- 数据集类 ------------------------------


class MyDataset(Dataset):
    '''加载数据集的类'''

    def __init__(self, root_dir, cls):
        '''初始化数据集,主要步骤如下:
        1.加载所有的csv表格'''

        self.df_l = []
        self.samples_all = 0  # 记录产生的sample time series总量
        self.samples_l = []  # 记录各个csv数据产生的sample time series长度
        for file in TQDM(os.listdir(root_dir), desc='加载%s数据' % cls):
            if not file.endswith('.csv'):  # 非csv数据跳过
                continue
            df_this = pd.read_csv(os.path.join(
                root_dir, file)).set_index('Time')[FEAT_SELECT]

            df_this = pd.DataFrame(SCALE.fit_transform(
                df_this), columns=df_this.columns)

            LEN = len(df_this)
            # 最后TROUBLE_LEN长度添加上标签1(其余是0),定义为列label
            label = np.concatenate([np.zeros(int(LEN - TROUBLE_LEN * 100)),
                                    np.ones(int(TROUBLE_LEN * 100))])
            df_this['label'] = label.astype(int)
            self.df_l.append(df_this)
            # 计算本csv数据能够产生的窗口数量
            window_num = LEN - WINDOW_LEN * 100 + 1
            self.samples_l.append(window_num)
            self.samples_all += window_num

    def __len__(self):
        return self.samples_all

    def __getitem__(self, idx):
        # 找出两个索引位置
        samples_cumsum = np.array(self.samples_l).cumsum()
        key1 = np.searchsorted(samples_cumsum, idx + 1)
        # 大于第一个cumsum 要减去前面的累加值，否则就是idx
        key2 = idx - samples_cumsum[key1 - 1] if key1 > 0 else idx

        # 通过key1找到对应的df
        target_df = self.df_l[key1]
        # 通过key2找到对应的value
        df_sample = target_df.iloc[key2: key2 + WINDOW_LEN * 100]

        # print(key1, key2, key2 + WINDOW_LEN * 100)

        X = df_sample.drop('label', axis=1).values
        y = df_sample['label'].values[-1]

        X = torch.Tensor(X)  # feat
        y = torch.tensor(y)  # label last

        return (X, y)  # shapeX= (3000,feat_num) shapey=(3000)


print('=================== 数据集构建 ===================')
BATCH_SIZE = 10  # batch数据的大小
train_dataset = MyDataset(root_dir=TRAIN_DIR, cls='训练')
test_dataset = MyDataset(root_dir=TEST_DIR, cls='测试')


# num_workers可能不支持
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=4)


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


train_datalen, test_datalen = len(train_dataset), len(test_dataset)


# -------------------------- 模型类 ------------------------------


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            batch_first=True, dropout=0.3)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(DEVICE),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(DEVICE))

    def forward(self, embeds):
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        tag_scores = self.hidden2tag(lstm_out[:, -1, :])  # 取最后一个时刻的标签值
        return tag_scores


# -------------------------- 模型训练 ------------------------------

EMBEDDING_DIM = len(FEAT_SELECT)
HIDDEN_DIM = 20
TAGSET_SIZE = 2
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE, BATCH_SIZE)
model.to(DEVICE)  # GPU/CPU
EPOCH = 20  # 训练周期
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# optimizer = optim.Adam(model.parameters(), lr=0.1)


def train_model():
    print('=================== 模型训练/测试 ===================')
    for epoch in range(EPOCH):
        # -----------------epoch 训练----------------------
        train_loss, train_pred, train_true = 0.0, [], []

        # for i, (X, y) in TQDM(enumerate(train_dataloader), desc='EPOCH-%d 训练' % epoch,
        #                       total=len(train_dataloader)):
        for i, (X, y) in enumerate(train_dataloader):
            model.train()  # 打开train模式
            X, y = X.to(DEVICE), y.to(DEVICE)  # GPU/CPU
            # Pytorch会累加梯度.我们需要在训练每个实例前清空梯度
            model.zero_grad()
            # 需要清空 LSTM 的隐状态,将其从上个实例的历史中分离出来.
            model.hidden = model.init_hidden()
            # 前向传播.
            tag_scores = model(X)
            # import ipdb
            # ipdb.set_trace()
            _, preds = tag_scores.max(1)
            # 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
            loss = loss_function(tag_scores, y)
            # backward
            loss.backward()
            # 防止梯度爆炸
            nn.utils.clip_grad_norm(model.parameters(), 5)
            # 优化器
            optimizer.step()
            # import ipdb
            # ipdb.set_trace()
            train_loss += loss.item()
            train_pred.extend(preds.tolist())
            train_true.extend(y.tolist())

            # 每N_loss次输出一下loss值
            N_loss = 5
            if(i + 1) % N_loss == 0:
                # print(train_pred, train_true)
                loss = train_loss / (N_loss * BATCH_SIZE)
                auc = roc_auc_score(train_true, train_pred)
                acc = accuracy_score(train_true, train_pred)
                # print('Train epoch:%d (%d/%d), Loss: %f, AUC: %f' %
                #       (epoch, i + 1, len(train_dataloader), loss, auc))
                print('Train epoch:%d(%d/%d),Loss:%f, AUC: %f, ACC=%f' %
                      (epoch, i + 1, len(train_dataloader), loss, auc, acc))
                train_loss, train_pred, train_true = 0.0, [], []

            # 每N_eval次做一下测试
            N_eval = 20
            if (i + 1) % N_eval == 0:
                # -----------------epoch 测试----------------------
                model.eval()  # 打开eval模式
                test_pred, test_true = [], []
                with torch.no_grad():  # 不改变梯度
                    for i, (X, y) in TQDM(enumerate(test_dataloader), desc='EPOCH-%d 测试' % epoch,
                                          total=len(test_dataloader)):
                        X, y = X.to(DEVICE), y.to(DEVICE)  # GPU/CPU
                        # 需要清空 LSTM 的隐状态,将其从上个实例的历史中分离出来.
                        model.hidden = model.init_hidden()
                        tag_scores = model(X)
                        _, preds = tag_scores.max(1)
                        test_pred.extend(preds.tolist())
                        test_true.extend(y.tolist())
                    auc = roc_auc_score(test_true, test_pred)
                    acc = accuracy_score(test_true, test_pred)

                    print(test_pred, test_true)
                    print('Test epoch:%d, AUC: %f, ACC=%f' % (epoch, auc, acc))


train_model()
