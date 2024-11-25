import argparse
import shutil
import os
import time
import logging
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from models.resnet_models import resnet19, resnet18
from models.VGG9_models import VGGSNN9
from models.VGG7_models import VGGSNN7
from models.MobilenetSNN import MBNETSNN, MBNETSNNWIDE, MBNETSNNWIDE_PostPool, MBNETSNN_NegQ, MBNETSNNWIDE_PostPool_NegQ, MBNETSNN_NegQ_LP
from models.MobilenetSNN import MBNETSNN
from models.methods import QBaseConv2d, QBaseLinear
from models.layers import LIFSpike
from models.t2c import LayerFuser, T2C
import shutil
import torch
import tabulate
import argparse
from functions import TET_loss, seed_all
import sys
sys.path.insert(1, '/home2/ahasssan/LV-Surrogate-Gradient-TE-SNN-VGG/dvsloader')
from dvsloader import dvs2dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lvth',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='if use learnable threshold (default: True)')
parser.add_argument('--lamb',
                    default=0.90,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--dataset',
                    default='ncars',
                    type=str,
                    metavar='N',
                    help='dataset')
parser.add_argument('--model',
                    default='MBNETSNN_LP',
                    type=str,
                    metavar='N',
                    help='model for training')
parser.add_argument('--save_path', 
                    type=str, 
                    default='./save/', 
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# for inference only
parser.add_argument('--neg',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='Threshold for negative membrane potential')
parser.add_argument('--membit',
                    default=2,
                    type=int,
                    help='quantization precision of the accumulated membrane potential')
parser.add_argument('--wbit',
                    default=4,
                    type=int,
                    help='quantization precision of the weights')
parser.add_argument('--thres',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='Potential threshold')
parser.add_argument('--tau',
                    default=0.5,
                    type=float,
                    metavar='N',
                    help='Leak factor')
parser.add_argument('--T',
                    default=30,
                    type=int,
                    metavar='N',
                    help='Time Stamps')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')

args = parser.parse_args()

activation = {}        
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def lr_schedule(epoch):
    if epoch >= 70:
        factor = 0.1
        if epoch >= 140:
            factor = 0.01
            if epoch >= 300:
                 factor = 0.001
    else:
        factor = 1.0
    return factor

def main_worker(local_rank, nprocs, args):

    if not os.path.exists('./save'):
        os.makedirs('./save')
    else:
        pass
    save_path=args.save_path
    log_file="training.log"

    # args = parser.parse_args()
    if not os.path.isdir(save_path):
         os.makedirs(save_path)
    
    # initialize terminal logger
    logger = logging.getLogger('training')
    if log_file is not None:
        fileHandler = logging.FileHandler(save_path+log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    args.local_rank = local_rank

    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23457',
                            world_size=args.nprocs,
                            rank=local_rank)

    if not args.fine_tune:
        load_names = None
    else:
        load_names = args.resume
    save_names = os.path.join(save_path, "checkpoint.pth.tar")

    if args.dataset == "dvscifar10":
        if args.T == 30:
            data_path="/home/jmeng15/data/dvs_cifar10_30steps/"
        elif args.T == 16:
            data_path = "/home2/jmeng15/data/dvs_cifar10_16steps/"
        elif args.T == 10:
            data_path="~/data/dvs_cifar10/"
        elif args.T == 8:
            data_path="/home/ahasssan/ahmed/LV-Surrogate-Gradient-TE-SNN-VGG/dvs_cifar10_8"
        din = [48, 48]
        train_loader, val_loader, num_classes = dvs2dataset.get_cifar_loader(data_path, batch_size=24, size=din[0])
    elif args.dataset == "ncars":
        if args.T == 30:
            data_path="/home2/jmeng15/data/ncars_pt/"
        elif args.T == 16:
            data_path="/home2/jmeng15/data/ncars_pt_t16/"
        din = [48, 48]
        train_loader, val_loader, num_classes = dvs2dataset.get_ncars_loader(data_path, batch_size=args.batch_size, size=din)
    elif args.dataset == "ibm_gesture":
        data_path = "/home2/jmeng15/data/ibm_gesture_pt"
        din = [48, 48]
        train_loader, val_loader, num_classes = dvs2dataset.get_cifar_loader(data_path, batch_size=24, size=din[0])
    
    if args.model == "MBNETSNN_LP":
        model = MBNETSNN_NegQ_LP(num_classes=num_classes, wbit=args.wbit, thres=args.thres, tau=args.tau)
    elif args.model == "MBNETSNN":
        model = MBNETSNN()
    elif args.model == "MBNETSNNWIDE_PostPool_NegQ":
        model = MBNETSNNWIDE_PostPool_NegQ()
    elif args.model == "VGGSNN7":
        model = VGGSNN7()
    elif args.model == "VGGSNN9":
        model = VGGSNN9(num_classes=num_classes)
    elif args.model == "resnet19":
        model = resnet19(num_classes=num_classes, T=args.T)
    model.T = args.T
    logger.info(model)


    if load_names != None:
        state_dict = torch.load(load_names)
        model.load_state_dict(state_dict, strict=False)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    cudnn.benchmark = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(
         train_loader)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_loader)

    logger = logger
    logger_dict = {}

    if args.evaluate:
        for n, m in model.named_modules():
            if isinstance(m, (QBaseConv2d, QBaseLinear, LIFSpike)):
                m.register_forward_hook(hook=get_activation(n))
        
        validate(val_loader, model, criterion, local_rank, args, logger, logger_dict)

        # t2c
        nn2c = T2C(model=model, swl=16, sfl=13, args=args)
        qnn = nn2c.nn2chip()
        print(qnn)
        validate(val_loader, qnn, criterion, local_rank, args, logger, logger_dict)
        nn2c.get_info(qnn)

        # save
        state = qnn.state_dict()
        filename = "t2c_model.pth.tar"
        path = os.path.join(args.save_path, filename)
        torch.save(state, path)

        # check the intermediate results
        for k, v in activation.items():
            print("l={}; shape={}".format(k, v.shape))
            fname = './layer_fm/'+ k +'.pt'
            torch.save(v.cpu(),fname)

        return

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        logger_dict["ep"] = epoch+1

        logger_dict["lr"] = scheduler.get_lr()[0]

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args, logger, logger_dict)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args, logger, logger_dict)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        t2 = time.time()
        print('Time elapsed: ', t2 - t1)
        print('Best top-1 Acc: ', best_acc1)

        logger_dict["Best_Accuracy"] = best_acc1
        # terminal log
        columns = list(logger_dict.keys())
        values = list(logger_dict.values())
        print_table(values, columns, epoch, logger)

        if is_best and save_names != None:
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), save_names)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args, logger, logger_dict):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    mean = 1.0 
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        images = images.float()
        target = target.cuda(local_rank, non_blocking=True)

        vthre = AverageMeter('vth', ':.4e')
        output = model(images)
        mean_out = torch.mean(output, dim=1)
        if not args.TET:
            loss = criterion(mean_out, target)
        else:
            loss = TET_loss(output, target, criterion, mean, args.lamb) ### Change mean to args.mean

        # measure accuracy and record loss
        if args.dataset == "ncars":
            acc1, = accuracy(mean_out, target, topk=(1, ))
        else:
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))
        
        cnt = 0
        for name, param in model.named_parameters():
            if 'thresh' in name:
                if cnt > 0:
                    vthre.update(param.item())
            cnt += 1

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        # reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        # top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        logger_dict["train_loss"] = losses.avg
        logger_dict["train_top1"] = top1.avg
        # logger_dict["train_top5"] = top5.avg
        logger_dict["avg_vth"] = vthre.avg
        
        # update regularization target
        if args.lvth:
            mean = vthre.avg
            # mean = 1.0


def validate(val_loader, model, criterion, local_rank, args, logger, logger_dict):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            images = images.float()
            if (images.size(0)<16):
                pass
            else:
                output = model(images)
                mean_out = torch.mean(output, dim=1)
                loss = criterion(mean_out, target)

                # measure accuracy and record loss
                if args.dataset == "ncars":
                    acc1, = accuracy(mean_out, target, topk=(1, ))
                else:
                    acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))
                # acc1, = accuracy(mean_out, target, topk=(1,))

                torch.distributed.barrier()

                reduced_loss = reduce_mean(loss, args.nprocs)
                reduced_acc1 = reduce_mean(acc1, args.nprocs)
                # reduced_acc5 = reduce_mean(acc5, args.nprocs)

                losses.update(reduced_loss.item(), images.size(0))
                top1.update(reduced_acc1.item(), images.size(0))
                # top5.update(reduced_acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

                logger_dict["valid_loss"] = losses.avg
                logger_dict["valid_top1"] = top1.avg
                logger_dict["valid_top5"] = top5.avg
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    lr = args.lr * (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

if __name__ == '__main__':
    main()
