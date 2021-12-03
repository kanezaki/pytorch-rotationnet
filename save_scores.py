import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--case', default='2', type=str,
                    help='viewpoint setup case (1 or 2)')
parser.add_argument('--input1', metavar='DIR', help='path of classes.txt')
parser.add_argument('--output_file', help="Output npy filename.")

best_prec1 = 0
vcand = np.load('vcand_case2.npy')  # (60,20)
nview = 20

with open(parser.parse_args().input1) as f:
    classes = f.readlines()
    classes = [f[:-1] for f in classes]
num_classes = len(classes)


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else:
            raise ("Finetuning not supported on this architecture yet")

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform):
        self.imgs_l_list = self.read_file(filename, transform)
        self.len = len(self.imgs_l_list)

    def __getitem__(self, i):
        index = i % self.len
        img, label = self.imgs_l_list[index]
        return img, label

    def __len__(self):
        data_len = len(self.imgs_l_list)
        return data_len

    def read_file(self, filename, transform):
        # args.input_file = os.path.expanduser(args.input_file)
        f = open(filename)
        line = f.readlines()
        imgs = []
        labels = []
        imgs_l_list = []
        for i in range(len(line)):
            content = line[i].split(' ')
            if len(content) == 1:
                content[0] = content[0][:-1]
            im_f = content[0].split('/', 1)[1]  # delete 'ModelNet40v2/' in file path
            img = Image.open(im_f)
            if transform is not None:
                img = transform(img)
            label = np.array(content[1].strip()).astype(np.int32)
            imgs.append(img)
            labels.append(label)
            imgs_l_list.append((img, label))

        return imgs_l_list


def main():
    global args, best_prec1, nview, vcand, num_classes
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    if args.case == '1':
        vcand = np.load('vcand_case1.npy')
        nview = 12
    elif args.case == '3':
        vcand = np.load('vcand_case3.npy')
        nview = 160

    if args.batch_size % nview != 0:
        print('Error: batch size should be multiplication of the number of views,', nview)
        exit()

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
 
    print("num_classes = '{}'".format(num_classes))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = FineTuneModel(model, args.arch, (num_classes + 1) * nview)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),  # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset = TorchDataset(filename=args.data,
                                 transform=transforms.Compose([
                                     # transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                 ]))
    print("=> there are {0} training data in {1}.".format(train_dataset.len, args.data))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # save checkpoint
        fname = 'checkpoints/save_scores_txt.pth.tar'
        if nview == 12:
            fname = 'checkpoints/save_scores_txt_case1.pth.tar'
        state = {'epoch': epoch + 1,
                 'arch': args.arch,
                 'state_dict': model.state_dict(),
                 'best_prec1': best_prec1,
                 'optimizer': optimizer.state_dict(),
                 }
        torch.save(state, fname)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        nsamp = int(input.size(0) / nview)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_ = torch.LongTensor(target.size(0) * nview)

        # compute output
        output = model(input_var)  # [8000, 40]
        num_classes = int(output.size(1) / nview) - 1
        output = output.view(-1, num_classes + 1)

        ###########################################
        # compute scores and decide target labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax(output)
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[:, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1) - 1).view(output_.size(1) - 1, -1))
        # reshape output matrix
        output_ = output_.view(-1, nview * nview, num_classes)
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose(1, 2, 0)

        # initialize target labels with "incorrect view label"
        for j in range(target_.size(0)):
            target_[j] = num_classes

        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros((vcand.shape[0], num_classes, nsamp))
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[j] = scores[j] + output_[vcand[j][k] * nview + k]

        # for each sample #n,
        # determine the best pose that maximizes the score for the target class (see Eq.(2))
        for n in range(nsamp):
            j_max = np.argmax(scores[:, target[n * nview], n])
            # assign target labels
            for k in range(vcand.shape[1]):
                target_[n * nview * nview + vcand[j_max][k] * nview + k] = target[n * nview]

            cls = np.argmax(scores[:, :, n]) % num_classes
            max_ang = np.argmax(scores[:, :, n]) // num_classes
            if epoch == (args.epochs - 1):  # save predicted labels and beat poses of samples for the last epoch
                with open(args.output_file, 'a') as f:
                    f.write(str(cls) + ' ' + str(max_ang) + '\n')
        ###########################################
        target_ = target_.cuda()
        target_var = torch.autograd.Variable(target_)

        # compute loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))


def save_checkpoint(state, is_best, filename, filename2):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename2)


class AverageMeter(object):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))


if __name__ == '__main__':
    main()
