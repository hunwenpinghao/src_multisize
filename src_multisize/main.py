# -*- coding: utf-8 -*-
"""
基于 PyTorch resnet50 实现的图片分类代码
原代码地址：https://github.com/pytorch/examples/blob/master/imagenet/main.py
可以与原代码进行比较，查看需修改哪些代码才可以将其改造成可以在 ModelArts 上运行的代码
在ModelArts Notebook中的代码运行方法：
（0）准备数据
大赛发布的公开数据集是所有图片和标签txt都在一个目录中的格式
如果需要使用 torch.utils.data.DataLoader 来加载数据，则需要将数据的存储格式做如下改变：
1）划分训练集和验证集，分别存放为 train 和 val 目录；
2）train 和 val 目录下有按类别存放的子目录，子目录中都是同一个类的图片
prepare_data.py中的 split_train_val 函数就是实现如上功能，建议先在自己的机器上运行该函数，然后将处理好的数据上传到OBS
执行该函数的方法如下：
cd {prepare_data.py所在目录}
python prepare_data.py --input_dir '../datasets/train_data' --output_train_dir '../datasets/train_val/train' --output_val_dir '../datasets/train_val/val'

（1）从零训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --pretrained True --seed 0

（2）加载已有模型继续训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --seed 0 --resume '../model_snapshots/epoch_0_2.4.pth'

（3）评价单个pth文件
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --arch 'resnet50' --num_classes 54 --seed 0 --eval_pth '../model_snapshots/epoch_5_8.4.pth'
"""
import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import numpy as np
from utils import GradualWarmupScheduler
import torch.hub as hub
from utils.augment import PowerPIL
from utils import LabelSmoothSoftmaxCEV2
from utils.ranger import Ranger, RangerQH, RangerVA
from utils.visdom import create_vis_plot, update_vis_plot, update_acc_plot, viz, FeatureExtractor

try:
    import moxing as mox
except:
    print('not use moxing')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import model as modelZoo
import pretrainedmodels
from utils.focalloss import FocalLoss_v2
from utils.CBAM import CBAM

from prepare_data import prepare_data_on_modelarts

resnext101_groups = ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl']
efficientnet_groups = ['efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names += ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50', 'se_resnext101', 'se_resnext152']
model_names += ['DPN26', 'DPN92']
model_names += ['se_resnext101_32x4d']
model_names += ['pnasnet5large']
model_names += ['se_resnext152']
model_names += efficientnet_groups
model_names += resnext101_groups

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=3, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--eval_pth', default='', type=str,
                    help='the *.pth model path need to be evaluated on validation set')
parser.add_argument('--pretrained', default=False, type=bool,
                    help='use pre-trained model or not')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--visdom', default=True, type=bool,
                    help='use visdom or not')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')
parser.add_argument('--local_data_root', default='/cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--test_data_url', default='', type=str, help='the test data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
parser.add_argument('--train_data', default='', type=str, help='the train data path')
parser.add_argument('--val_data', default='val_wwd', type=str, help='the val data path')
parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str,
                    help='a path which contain config.json and customize_service.py, '
                         'if it is set, these two scripts will be copied to {train_url}/model directory')
best_acc1 = 0


def main():
    args, unknown = parser.parse_known_args()
    print('use args:', args)
    #args = prepare_data_on_modelarts(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpus = list(range(len(args.gpu.split(','))))
    else:
        gpus = [0]  # [1,2]

    # if args.arch not in model_names:
    #     raise NotImplementedError('Other optimizer is not implemented')
    # # elif args.arch == 'DPN26' or 'DPN92':
    #     # from model import MultiModalNet
    #     # model = MultiModalNet("se_resnext101_32x4d","dpn26",0.5)
    # else:
    #     Net = getattr(modelZoo, args.arch)
    #     model = Net(num_classes=args.num_classes)

    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if args.pretrained:
        if args.arch == 'se_resnext101_32x4d':
            print('=> creating model {}'.format(args.arch))
            Net = getattr(modelZoo, args.arch)
            model = Net(args.arch, 0.5, args.num_classes) # here only use 'se_resnext101_32x4d'
        elif args.arch == 'pnasnet5large' or args.arch == 'se-resnext152':
            print('=> creating model {}'.format(args.arch))
            model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
            model.last_linear = nn.Linear(model.last_linear.in_features, args.num_classes)
        elif args.arch in resnext101_groups:
            print('=> creating model {}'.format(args.arch))
            model = hub.load('facebookresearch/WSL-Images', args.arch)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(2048, 1024), nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024, 54))
        elif args.arch in efficientnet_groups:
            print('=> creating model {}'.format(args.arch))
            # model = modelZoo.EfficientNet_CBAM(args.arch)

            Net = getattr(modelZoo, 'MultiNet')
            model = Net(args.arch)

            # Net = getattr(modelZoo, 'EfficientNet')
            # model = Net.from_pretrained(args.arch)
            # model._fc = nn.Linear(model._fc.in_features, args.num_classes)

            # model._fc = nn.Sequential(
            #     nn.BatchNorm1d(model._fc.in_features),
            #     nn.Linear(model._fc.in_features, 256),
            #     nn.LeakyReLU(inplace=True),
            #     nn.Dropout(0.4),
            #     nn.Linear(256, args.num_classes)
            # )
        else:
            print('=> creating model {}'.format(args.arch))
            Net = getattr(models, args.arch)
            model = Net(pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = FocalLoss_v2(num_class=args.num_classes).cuda(args.gpu)
    # criterion = LabelSmoothSoftmaxCEV2(lb_smooth=0.1, lb_ignore=-100)
    # criterion.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = RangerVA(model.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-6)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if os.path.exists(args.resume) and (not os.path.isdir(args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume.startswith('/cache/tmp/'):
                os.remove(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_local, args.train_data)
    valdir = os.path.join(args.data_local, args.val_data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    # ImageFolder类会将traindir目录下的每个子目录名映射为一个label id，然后将该id作为模型训练时的标签
    # 比如，traindir目录下的子目录名分别是0~53，ImageFolder类将这些目录名当做class_name，再做一次class_to_idx的映射
    # 最终得到这样的class_to_idx：{"0": 0, "1":1, "10":2, "11":3, ..., "19": 11, "2": 12, ...}
    # 其中key是class_name，value是idx，idx就是模型训练时的标签
    # 因此我们在保存训练模型时，需要保存这种idx与class_name的映射关系，以便在做模型推理时，能根据推理结果idx得到正确的class_name




    # if args.eval_pth != '':
    #     if mox.file.exists(args.eval_pth) and (not mox.file.is_directory(args.eval_pth)):
    #         if args.eval_pth.startswith('s3://'):
    #             model_name = args.eval_pth.rsplit('/', 1)[1]
    #             mox.file.copy(args.eval_pth, '/cache/tmp/' + model_name)
    #             args.eval_pth = '/cache/tmp/' + model_name
    #         print("=> loading checkpoint '{}'".format(args.eval_pth))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.eval_pth)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.eval_pth, map_location=loc)
    #         if args.eval_pth.startswith('/cache/tmp/'):
    #             os.remove(args.eval_pth)
    #
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.eval_pth, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.eval_pth))
    #
    #     validate(val_loader, model, criterion, args)
    #     return

    if args.visdom:
        vis_title = 'PyTorch on DCNN'
        vis_legend = ['Train Loss', 'Val Loss']
        vis_legend_1 = ['Acc-Top1', 'Acc-Top5']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
        acc_plot = create_vis_plot('Iteration', 'Acc', vis_title, vis_legend_1)
        plots = [iter_plot, epoch_plot, acc_plot]

    for epoch in range(args.start_epoch, args.epochs):
        # load data ========================================================
        switch_freq = 10
        if (epoch // switch_freq) % 3 == 0:
            input_size = 256 # bz=18
            args.batch_size = 18
        elif (epoch // switch_freq) % 3 == 1:
            input_size = 380 # bz=8
            args.batch_size = 8
        elif (epoch // switch_freq) % 3 == 2:
            input_size = 456 # bz=6
            args.batch_size = 6
        print('input_size, bz:', input_size, args.batch_size)

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                # PowerPIL(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        idx_to_class = OrderedDict()
        for key, value in train_dataset.class_to_idx.items():
            idx_to_class[value] = key

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # ===========================================================================
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, plots)

        # evaluate on validation set
        if (epoch + 1) % args.print_freq == 0:
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = False
            best_acc1 = max(acc1.item(), best_acc1)
            pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                         % (str(epoch + 1), str(round(acc1.item(), 3))))
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'idx_to_class': idx_to_class
                }, is_best, pth_file_name, args)

    if args.epochs >= args.print_freq:
        save_best_checkpoint(best_acc1, args)


def train(train_loader, model, criterion, optimizer, epoch, args, plots):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    epoch_size = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, out2 = model(images)
        # viz.images(images, win='img src', opts={'title': 'headmaps'})
        # viz.images(images_att, win='img attention', opts={'title': 'headmaps'})
        loss_1 = criterion(output, target)
        loss_2 = criterion(out2, target)
        loss = loss_1 + 0.1 * loss_2

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.visdom:
                iter_plot = plots[0]
                epoch_plot = plots[1]
                acc_plot = plots[2]
                if epoch != 0:
                    update_vis_plot(epoch, loss.item(), loss.item(), epoch_plot, None,
                                    'append', args.print_freq)

                update_vis_plot(epoch * epoch_size + i, loss.item(), loss.item(), iter_plot, epoch_plot, 'append')
                update_acc_plot(epoch * epoch_size + i, top1.avg, top5.avg, acc_plot, 'append')

                # with open('model.txt', 'w') as f:
                #     f.write(str(model))
                #
                # feature = images
                # feature = feature.cuda(args.gpu, non_blocking=True)
                # for name,module in model._modules['module']._modules.items():
                #     feature = module(feature)
                #     if name == '_bn0':
                #         break
                #
                # # for i in range(len(model._modules['module']._modules['_blocks'])):
                # for i in range(6):
                #     for name,module in model._modules['module']._modules['_blocks'][i]._modules.items():
                #         feature = module(feature)
                #         if name == '_bn2':
                #             feature_show = torch.sum(feature[0], 0)
                #             viz.heatmap(feature_show.data, 'feature{}'.format(i))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, out2 = model(images)
            loss = criterion(output, target) + criterion(out2, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename, args):
    if not is_best:
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            mox.file.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)


def save_best_checkpoint(best_acc1, args):
    best_acc1_suffix = '%s.pth' % str(round(best_acc1, 3))
    print('best_acc1_suffix:', best_acc1_suffix)
    pth_files = os.listdir(args.train_local)
    pthfile = ''
    for pth_name in pth_files:
        if pth_name.endswith(best_acc1_suffix):
            pthfile = pth_name
            break

    # mox.file可兼容处理本地路径和OBS路径
    if not os.path.exists(os.path.join(args.train_url, 'model')):
        os.makedirs(os.path.join(args.train_url, 'model'))
    print('pthfile:', pthfile)
    os.system('cp {} {}'.format(os.path.join(args.train_local, pthfile), os.path.join(args.train_url, 'model/model_best.pth')))
    os.system('cp {} {}'.format(os.path.join(args.deploy_script_path, 'config.json'),
                  os.path.join(args.train_url, 'model/config.json')))
    os.system('cp {} {}'.format(os.path.join(args.deploy_script_path, 'customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py')))
    if os.path.exists(os.path.join(args.train_url, 'model/config.json')) and \
            os.path.exists(os.path.join(args.train_url, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()