import argparse
import os
import os.path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_vgg16 import ClothingAttributeNet
from dataset import DatasetProcessing
import pdb
import time
import torchvision
import torchvision.transforms as transforms
import shutil


from sklearn.metrics import precision_score

import numpy as np

global best_mean_AP 
best_mean_AP =  0

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Fashion Clothing attribut prediction')
    parser.add_argument('--model', type=str, default='alexnet', metavar='M',
                        help='model name')
    parser.add_argument('--save_path', type=str, default='./snapshot/', metavar='PATH',
                        help='save path name')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 90)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    clothing_attributes_num = (11,4,6,4,5,3,2,2,2,3)
    clothing_attributes_name = ['color', 'style', 'collar', 'styleofcolor', 'styleofsleeve',
                                'lengthofsleeve','zip', 'belt', 'button', 'lengthofwhole']
    model = ClothingAttributeNet(args.model, clothing_attributes_num)
    print(model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_mean_AP = checkpoint['best_mean_AP']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_mean_AP = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # state_dict = model.state_dict()
    # for k, v in state_dict.items():
    #     print(k)



    data_path_train = '/home/zhangsy/zsy/clotthing_attributes/datasets/'
    img_path_train = 'detected_womanclothes/train_clothes/'
    img_filename_train ='woman_clothes_names_train.txt'
    attribute_filename_train = 'woman_clothes_attributes_train.txt'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


    train_data = DatasetProcessing(data_path_train,img_path_train,img_filename_train,attribute_filename_train,train_transforms)


    train_loader = torch.utils.data.DataLoader(train_data, 
                    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)


    data_path_test = '/home/zhangsy/zsy/clotthing_attributes/datasets/'
    img_path_test = 'detected_womanclothes/test_clothes/'
    img_filename_test ='woman_clothes_names_test.txt'
    attribute_filename_test = 'woman_clothes_attributes_test.txt'



    test_transforms=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])


    test_data = DatasetProcessing(data_path_test,img_path_test,img_filename_test,attribute_filename_test,test_transforms)


    test_loader = torch.utils.data.DataLoader(test_data, 
                    batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)



    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_params = [{'params': model.backbone.parameters(), 'lr': args.lr},
                    {'params': model.task_attention.parameters(), 'lr': args.lr * 10},
                    {'params': model.attribute_feature.parameters(), 'lr': args.lr * 10},
                    {'params': model.attribute_classifier.parameters(), 'lr': args.lr * 10}]

    optimizer = torch.optim.SGD(train_params,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_mean_AP = 0
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        AP = validate(test_loader, model, criterion, args)
        mean_AP = np.mean(AP)
        # print average precision (AP) on each clothing attribute
        for i in range(10):
            print(i,'.',clothing_attributes_name[i],':',AP[i])
        print('mean Average Precision (mAP):',mean_AP)

        # remember best acc@1 and save checkpoint
        is_best = mean_AP > best_mean_AP
        best_mean_AP = max(mean_AP, best_mean_AP)


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_mean_AP': best_mean_AP,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model.cuda()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(images)

        
        loss = 0 
        for j in range(len(output)):
            loss += criterion(output[j], target[:,j])
        
        losses.update(loss.item(), images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            progress.display(i)
            #print('lr:',optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'])



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, ],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    all_output = [[] for i in range(10)]
    all_target = [[] for i in range(10)]
    
    acc1 = [[] for i in range(10)]
    top1 = [[] for i in range(10)]
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            
            output = model(images)
            
            loss = 0 
            for j in range(len(output)):
                loss += criterion(output[j], target[:,j])
    
                if i==0:
                    all_output[j] = output[j].cpu()
                    all_target[j] = target[:,j].cpu()
                else:
                    all_output[j] = torch.cat((all_output[j],output[j].cpu()), dim=0)
                    all_target[j] = torch.cat((all_target[j],target[:,j].cpu()), dim=0)

            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


    
    ### compute average precision and mean average precision
    AP = [[] for i in range(10)]
    for k in range(10):
        y_pred = all_output[k].detach().numpy()
        y_pred_index = np.argmax(y_pred,axis=1)
        y_true = all_target[k].detach().numpy()
        
        AP[k] = precision_score(y_true, y_pred_index, average="macro",zero_division=0)

   
    return AP


def save_checkpoint(state, is_best, save_path='./snapshot/', filename='checkpoint.pth.tar'):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


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
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    
    optimizer.param_groups[0]['lr'] = lr

    for i in range(1, len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr * 10


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