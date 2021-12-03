import argparse
import os
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

parser = argparse.ArgumentParser(description='PyTorch Pose Estimation')
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
vcand = np.load('vcand_case2.npy')
nview = 20

# Use class.txt to get classes and calculate num_classes
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
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
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

        # # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

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
        self.imgs_vids_list = self.read_file(filename, transform)
        self.len = len(self.imgs_vids_list)

    def __getitem__(self, i):
        index = i % self.len
        img, vid = self.imgs_vids_list[index]
        return img, vid

    def __len__(self):
        data_len = len(self.imgs_vids_list)
        return data_len

    def read_file(self, filename, transform): # different from save_scores.py, read filenames and vids(not labels)
        # filename = os.path.expanduser(filename)
        f = open(filename)
        line = f.readlines()
        imgs = []
        vids = []
        imgs_vids_list = []
        for i in range(len(line)):
            content = line[i].split(' ')
            if len(content) == 1:
                content[0] = content[0][:-1]
            im_f = content[0]
            img = Image.open(im_f)
            if transform is not None:
                img = transform(img)
            vid = np.array(content[-1].strip()).astype(np.int32)
            imgs.append(img)
            vids.append(vid)
            imgs_vids_list.append((img, vid))

        return imgs_vids_list


def main():
    global args, best_prec1, nview, vcand
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.case == '1':
        vcand = np.load('vcand_case1.npy')
        nview = 12
    elif args.case == '3':
        vcand = np.load('vcand_case3.npy')
        nview = 160
        
    '''
    if args.batch_size % nview != 0:
        print('Error: batch size should be multiplication of the number of views,', nview)
        exit()
    '''

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

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch'] # If this is not commented, the epoch will start with the 200 of the last training.
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    test_dataset = TorchDataset(filename=args.data,
                                transform=transforms.Compose([
                                    # transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                ]))
    print("=> there are {0} test data in {1}.".format(test_dataset.len, args.data))

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    '''
    # I think in the actual application, multiple-view images should belong to the same sample, 
    # so I commented these lines and line 265-271.
    sorted_imgs = sorted(test_loader.dataset.imgs_l_list)
    test_nsamp = int(len(sorted_imgs) / nview)
    '''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            test_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)
        '''
        inds = np.zeros((nview, test_nsamp)).astype('int')
        inds[0] = np.random.permutation(range(test_nsamp)) * nview
        for i in range(1, nview):
            inds[i] = inds[0] + i
        inds = inds.T.reshape(nview * test_nsamp)
        test_loader.dataset.imgs = [sorted_imgs[i] for i in inds]
        test_loader.dataset.samples = test_loader.dataset.imgs
        '''
        # test for one epoch 
        pose_estimation(test_loader, model)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))


def pose_estimation(test_loader, model):
    global num_classes, classes
    num_show_train_images = 10

    model.eval()

    for i, (input, vid) in enumerate(test_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input, volatile=True)

        output = model(input_var)  # [batch_size, 820]
        
        # The following is the same as classify_npyfile_case2_with_pose.py in caffe version.
        if (len(output) > nview) or (len(vid) > nview):
            print('Please input the info. of a single sample.')
            exit()
        
        scores = np.ones((nview, len(output[0])))
        output = output.cpu().detach().numpy()
        
        for i in range(len(vid)):
            scores[vid[i]] = output[i]

        for i in range(0, len(scores)):
            for j in range(0, nview):
                for k in range(0, num_classes):
                    scores[i][j * (num_classes + 1) + k] = scores[i][j * (num_classes + 1) + k] / scores[i][
                        j * (num_classes + 1) + num_classes]
                scores[i][j * (num_classes + 1) + num_classes] = 0
        num_classes = (num_classes + 1)

        s = np.ones(num_classes * vcand.shape[0])
        for i in range(vcand.shape[0]):
            for j in range(num_classes):
                for k in range(nview):
                    s[i * num_classes + j] = s[i * num_classes + j] * scores[vcand[i][k]][k * num_classes + j]
                    
        cls = np.argmax(s) % num_classes
        max_ang = np.argmax(s) // num_classes
        print('***********************')
        print('** predicted:', classes[cls], '**')
        print('***********************')

        # show image
        f = open(args.data)
        lines = f.readlines()
        imfiles = [v.split(' ')[0] for v in lines]

        images = []
        for i in imfiles:
            images.append(np.array(Image.open(i)))
        images = np.concatenate(images, axis=0)
        Image.fromarray(images).show()
        Image.fromarray(images).save("align_images/inputs.png") # save the image in order to view it directly in PyCharm

        # show reference images
        f = open('reference_poses/reference_poses_' + classes[cls] + '.txt') # oredered images, home/guan/pytorch-rotationnet/
        lines = f.readlines()
        f.close()
        imfiles = [v[:-1].split('/', 1)[1] for v in lines]
        images_multi = []
        for i in range(len(vid)):
            images = []
            for n in range(num_show_train_images):
                predicted_view_id = vcand[max_ang][vid[i]]  
                idx = n * nview + predicted_view_id
                images.append(np.array(Image.open(imfiles[idx])))

            images = np.concatenate(images, axis=1)
            images_multi.append(images)
        images = np.concatenate(images_multi, axis=0)
        Image.fromarray(images).show()
        Image.fromarray(images).save("align_images/" + str(classes[cls]) + ".png")


if __name__ == '__main__':
    main()
