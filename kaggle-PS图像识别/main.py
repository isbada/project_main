import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
from torch.nn.functional import softmax

# ===================================== 全局变量
# PIL 2 Tensor转化器
data_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_dir, unlabel_dir = './raw/training', './raw/unknown'


# ===================================== 具体步骤

# --------------1 读取所有的(label/unlabel)数据,将数据划分为train val test
#  读取所有的有标签数据,作为train val
all_labeled = datasets.ImageFolder(label_dir, data_transform)
class_names = all_labeled.classes  # 所有类的名称

#  读取所有的无标签数据,作为test
all_unlabeled = datasets.ImageFolder(unlabel_dir, data_transform)

# # 采样减小all_labeled数据量
# left_num = int(len(all_labeled) * 0.1)
# remove_num = len(all_labeled) - left_num
# _, all_labeled = torch.utils.data.random_split(
#     all_labeled, [remove_num, left_num])


def generate_dataloader(train_pct=0.9):
    '''随机划分训练和验证集的函数，返回dataloaders, dataset_sizes'''
    global all_labeled
    train_size = int(len(all_labeled) * train_pct)
    val_size = len(all_labeled) - train_size

    image_datasets = {}
    image_datasets['train'], image_datasets['val'] = torch.utils.data.random_split(
        all_labeled, [train_size, val_size])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # print(image_datasets['val'].indices)

    return dataloaders, dataset_sizes


# ---------------- 2.可视化一些处理后的数据（可选步骤）

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


dataloaders, dataset_sizes = generate_dataloader()
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
plt.show()


# ---------------- 3.模型训练
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    '''模型训练集成函数'''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都重新划分训练/测试数据，让模型尽可能的多看数据
        dataloaders, dataset_sizes = generate_dataloader()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        outputs = model(inputs)
                    except:
                        import traceback
                        traceback.print_exc()
                        import ipdb
                        ipdb.set_trace()
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# 采用微调预训练卷积网络参数的方法进行训练

# 加载预训练resnet18模型
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# num_ftrs = 8192
# 重置最终的全连接层
model_ft.fc = nn.Linear(num_ftrs, 2)
# loss函数
criterion = nn.CrossEntropyLoss()
# 参数优化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 优化器的调节器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 可能进行GPU运算
model_ft = model_ft.to(device)
# 训练和评估
model_ft = train_model(model_ft, criterion, optimizer_ft,
                       exp_lr_scheduler, num_epochs=25)


# ---------------- 6.使用模型预测测试数据

with torch.no_grad():
    test_tensors = torch.cat(
        list(map(lambda t: t[0].unsqueeze(0), all_unlabeled)))
    test_tensors = test_tensors.to(device)
    # 预测fake的结果
    print('预测fake的概率....')
    prob_fake = softmax(model_ft(test_tensors), dim=1)[:, 0].tolist()
    # 获取测试图片名称
    ids = list(map(lambda t: t[0].split('/')
                   [-1].split('.')[0], all_unlabeled.imgs))
    # 生成提交的csv和保存
    df_submit = pd.DataFrame({'ID': ids, 'fake': prob_fake})

    print('sample_submission.csv生成....')
    df_submit.to_csv('sample_submission.csv', index=None)
