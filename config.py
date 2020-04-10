# Learning Loss for Active Learning

__doc__ = """training configs"""

## dataset
CIFAR10_PATH = '/nfs/xs/Datasets/CIFAR10'
CIFAR100_PATH = '/nfs/xs/Datasets/CIFAR100'

NUM_TRAIN = 50000  # N
NUM_VAL = 50000 - NUM_TRAIN  # 0？
BATCH = 128  # B
SUBSET = 10000  # M
ADDENDUM = 1000  # K

# hyper
MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda

TRIALS = 3
CYCLES = 10

# optimizer
LR = 0.1
MOMENTUM = 0.9
WDECAY = 5e-4

# After 120 epochs, stop the gradient from the loss prediction module propagated to the target model
EPOCH = 200  # epoch for target net
EPOCHL = 120  # epoch for loss net
MILESTONES = [160]  # 到这里 lr=0.01

''' CIFAR-10 | ResNet-18 | 93.6%
NUM_TRAIN = 50000 # N
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = NUM_TRAIN # M
ADDENDUM  = NUM_TRAIN # K

MARGIN = 1.0 # xi
WEIGHT = 0.0 # lambda

TRIALS = 1
CYCLES = 1

EPOCH = 50
LR = 0.1
MILESTONES = [25, 35]
EPOCHL = 40

MOMENTUM = 0.9
WDECAY = 5e-4
'''
