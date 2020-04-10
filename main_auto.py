'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)

visualize status with tensorboard
'''

# Python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.1"

import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler  # 随机采样子集
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
from tqdm import tqdm

# Custom
from data.sampler import SubsetSequentialSampler  # 序列采样子集
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from utils import *
import config

tag = 'res18_cifar10'

# exp
exp = f'{tag}_{get_curtime()}'
exp_dir = f'exp/{exp}'
os.makedirs(exp_dir, exist_ok=True)
dump_json(cfg2dict(config), out_path=f'{exp_dir}/config.json')
# tensorboard / logs
writer = SummaryWriter(log_dir=os.path.join('runs', exp))
sys.stdout = Logger(f'{exp_dir}/run.log', sys.stdout)
# output model
ckpt_dir = os.path.join('output', exp)
os.makedirs(ckpt_dir, exist_ok=True)
print('exp:', exp)

# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_train = CIFAR10(CIFAR10_PATH, train=True, download=True, transform=train_transform)
cifar10_unlabeled = CIFAR10(CIFAR10_PATH, train=True, download=True, transform=test_transform)
cifar10_test = CIFAR10(CIFAR10_PATH, train=False, download=True, transform=test_transform)


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    Args:
        input: loss_module input
        target: target model output loss
        margin:
        reduction:
    """
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    # [l_1 - l_B, l_2 - l_B-1, ... , l_B/2 - l_B/2+1], where batch_size = B = 128
    # 首尾相减，再取1半，恰好 B/2
    input = (input - input.flip(0))[:len(input) // 2]  # B/2
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()  # 作为 label

    # 公式(2) 定义的 s.t. 约束条件
    # if target > 0: sign = 1, one = 1
    #          else: sign = 0, one = -1
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved as B/2 at line 76
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        raise NotImplementedError()

    return loss


def compute_acc(scores, labels):
    _, preds = torch.max(scores.data, 1)
    acc = (preds == labels).sum().item() / labels.size(0)

    return acc * 100


def record_lr(optimizers, epoch):
    lr_backbone = get_lr(optimizers['backbone'])
    lr_module = get_lr(optimizers['module'])
    writer.add_scalars(f'{prefix}/lr', {
        'backbone': lr_backbone,
        'module': lr_module
    }, global_step=epoch)


def train_epoch(models, criterion, optimizers, dataloaders,
                epoch, epoch_loss):
    """
        epoch:      current epoch in train()
        epoch_loss: loss total epoch: EPOCHL = 120
    """
    models['backbone'].train()
    models['module'].train()

    train_backbone_loss, train_module_loss = AverageMeter(), AverageMeter()
    train_loss, train_acc = AverageMeter(), AverageMeter()

    t = tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train']))  # leave=False 刷新
    t.set_description(f"Epoch: {epoch}")

    for data in t:
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # score: [10,1], features: [layer1,2,3,4]
        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)  # reduction='none'

        if epoch > epoch_loss:
            # After 120 epochs
            # stop the gradient from the loss prediction module propagated to the target model.
            # detach features from backbone; the loss won't propagated ...
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)  # FC(cat(features))
        pred_loss = pred_loss.view(pred_loss.size(0))  # B=128

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)  # 公式(2)(3)

        # total loss
        loss = m_backbone_loss + WEIGHT * m_module_loss  # WEIGHT=1, 公式(1)

        loss.backward()

        optimizers['backbone'].step()
        optimizers['module'].step()

        train_backbone_loss.update(m_backbone_loss)
        train_module_loss.update(m_module_loss)
        train_loss.update(loss)
        train_acc.update(compute_acc(scores, labels))

    return {
        'backbone_loss': train_backbone_loss.avg,
        'module_loss': train_module_loss.avg,
        'total_loss': train_loss.avg,
        'acc': train_acc.avg
    }


@torch.no_grad()
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    test_loss, test_acc = AverageMeter(), AverageMeter()

    for (inputs, labels) in dataloaders[mode]:
        inputs = inputs.cuda()
        labels = labels.cuda()

        scores, _ = models['backbone'](inputs)

        loss = F.cross_entropy(scores, labels, reduction='mean')
        test_loss.update(loss)
        test_acc.update(compute_acc(scores, labels))

    return test_loss.avg, test_acc.avg


def train(models, criterion, optimizers, schedulers,
          dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in range(1, num_epochs + 1):
        record_lr(optimizers, epoch)
        train_status = train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

        # update optimizer lr
        # after optimizer.step(), in case skipping the 1st val of lr_schedule
        schedulers['backbone'].step()
        schedulers['module'].step()

        # Save a checkpoint
        if epoch % 5 == 0:  # 每 5 个 epoch，check 一下
            test_loss, acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                }, f'{ckpt_dir}/{tag}.pth')
            print('Epoch: {}, Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(epoch, acc, best_acc))

            writer.add_scalars(f'{prefix}/train_loss', {
                'backbone': train_status['backbone_loss'],
                'module': train_status['module_loss'],
                'total': train_status['total_loss'],
            }, global_step=epoch)
            writer.add_scalar(f'{prefix}/test_loss', test_loss, global_step=epoch)
            writer.add_scalars(f'{prefix}/accuracy', {
                'train': train_status['acc'],
                'test': acc
            }, global_step=epoch)

    print('>> Finished.')


@torch.no_grad()
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    # store batchs pred_loss
    uncertainty = torch.tensor([]).cuda()
    for (inputs, labels) in unlabeled_loader:
        inputs = inputs.cuda()
        # labels = labels.cuda()  # not use

        scores, features = models['backbone'](inputs)
        pred_loss = models['module'](features)
        # gt_pred_loss = criterion(scores, labels) # ground truth loss
        # 使用 gt_loss 比不上 loss_module 得到的相对 loss
        pred_loss = pred_loss.view(pred_loss.size(0))

        uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    seeds = [10, 20, 30]

    for trial in range(TRIALS):  # 减少实验随机性
        # 随机采样 L0/U0
        random.seed(seeds[trial])
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]  # L0, K=ADDENDUM=1,000
        unlabeled_set = indices[ADDENDUM:]  # U0
        random.seed()

        train_loader = DataLoader(cifar10_train, batch_size=BATCH,  # 从 cifar10_train 选出 labeset
                                  sampler=SubsetRandomSampler(labeled_set),  # 再次随机 labeled_set 顺序
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18 = resnet.ResNet18(num_classes=10).cuda()
        loss_module = lossnet.LossNet().cuda()
        models = {
            'backbone': resnet18,  # target
            'module': loss_module  # loss
        }

        # Active learning cycles
        for cycle in range(CYCLES):  # 10
            prefix = f'trial{trial + 1}/cycle{cycle + 1}'

            # criterion, optimizer and scheduler (re)initialization 重新初始化
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,  # 只有 h 之后的参数
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)  # [160]
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            writer.add_scalars(f'trial{trial + 1}/dataset', {
                'label': len(labeled_set),
                'unlabel': len(unlabeled_set)
            }, global_step=cycle)

            # train 200 epochs and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            _, acc = test(models, dataloaders, mode='test')

            writer.add_scalar(f'trial{trial + 1}/cycle_acc', acc, global_step=cycle)

            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)  # U0
            subset = unlabeled_set[:SUBSET]  # 10000, idxs

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),  # 序列采样，反正随机过了，而且只是推理需要
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order, loss 升序排列
            arg = np.argsort(uncertainty)  # tensor

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())  # 取出 loss 最大的 K 个
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
            # subset 取完 ADDENDUM 剩下的 + unlabeled_set 取完 subset 剩下的

            # new train dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({
            'trial': trial + 1,
            'state_dict_backbone': models['backbone'].state_dict(),
            'state_dict_module': models['module'].state_dict()
        }, f'{ckpt_dir}/{tag}_trial{trial + 1}.pth')
