import time


def cfg2dict(a):
    # 常量定义的 py module 转成 dict
    return {
        key: val for key, val in a.__dict__.items()
        if '__' not in key  # 非内部参数
    }


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
