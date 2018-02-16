import argparse
import os
import shutil
from collections import OrderedDict

import numpy as np
import sys
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pprint import pformat
from tqdm import tqdm, trange

# Available PyTorch models for ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class CosineDistance(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineDistance, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return - nn.functional.cosine_similarity(x1, x2, self.dim, self.eps)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        base_dir = os.path.dirname(filename)
        best_filename = os.path.join(base_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(data, model, criterion, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X, Y = sample_batched
        if args.embeddings:
            _, Y = Y  # take the target embedding from (idx, embedding) couple

        if args.cuda:
            X = X.cuda()
            Y = Y.cuda(async=True)

        X = torch.autograd.Variable(X, requires_grad=False)
        Y = torch.autograd.Variable(Y, requires_grad=False)

        # forward and backward pass
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = criterion(Y_hat, Y).mean()
        loss.backward()
        optimizer.step()

        loss = loss.data[0]
        progress_bar.set_postfix(dict(loss=loss))
        avg_loss += loss

        if batch_idx % args.log_interval == 0:
            avg_loss /= args.log_interval
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss), file=args.log,
                flush=True)
            avg_loss = 0.0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    vals, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k.mul_(100.0 / batch_size)
        res.append(correct_k.data[0])
    return res


def evaluate(data, model, criterion, epoch, embeddings, args):
    model.eval()

    if args.embeddings:
        if args.cuda:
            embeddings = embeddings.cuda()
        embeddings = torch.autograd.Variable(embeddings, requires_grad=False)

    avg_acc1 = 0.0
    avg_acc5 = 0.0
    avg_loss = 0.0
    n_samples = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X, Y = sample_batched

        if args.embeddings:
            Y, Z = Y  # Z is the target embedding

        if args.cuda:
            X = X.cuda()
            Y = Y.cuda(async=True)

            if args.embeddings:
                Z = Z.cuda(async=True)

        X = torch.autograd.Variable(X, volatile=True)
        Y = torch.autograd.Variable(Y, volatile=True)

        if args.embeddings:
            Z = torch.autograd.Variable(Z, volatile=True)

        Y_hat = model(X)

        target = Z if args.embeddings else Y
        loss = criterion(Y_hat, target).mean()
        loss = loss.data[0]

        # compute cosine (dot-product only) against all embeddings
        if args.embeddings:
            Y_hat = Y_hat.mm(embeddings.t())

        acc1, acc5 = accuracy(Y_hat, Y, topk=(1, 5))

        avg_loss += loss
        avg_acc1 += acc1
        avg_acc5 += acc5
        n_samples += 1

        progress_bar.set_postfix(OrderedDict(
            loss='{:6.4f}'.format(loss),
            acc1='{:5.2f}%'.format(acc1),
            acc5='{:5.2f}%'.format(acc5)
        ))

    avg_loss /= n_samples
    avg_acc1 /= n_samples
    avg_acc5 /= n_samples

    print('Test Epoch {}: Loss = {:.5} Acc@1 = {:.3}% Acc@5 = {:.3}%'.format(epoch, avg_loss, avg_acc1, avg_acc5),
          file=args.log, flush=True)

    return avg_loss


def adapt(model, embed_size, args):

    if args.fix_weights:
        for param in model.parameters():
            param.requires_grad = False
            
    def make_new_layers(I, E, N):
        assert N >= 1 and type(N) is int, "N must be an integer >= 1"
        
        if N == 1:
            return nn.Linear(I, E)
            
        if N == 2:
            M = round((I * 1000) / (I + E))
            return nn.Sequential(nn.Linear(I, M), nn.ReLU(), nn.Linear(M, E))
            
        # CASE N > 2:
        # I*M + (N-2)*(M*M) + M*E = I * 1000, solve for M:
        M = round((np.sqrt( (I + E)**2 + 4*(N-2)*1000*I ) - (I + E)) / (2*(N-2)))
        new_layers = [nn.Linear(I, M), nn.ReLU()]
        for _ in range(N-2):
            new_layers.extend([nn.Linear(M, M), nn.ReLU()])
        new_layers.append(nn.Linear(M, E))
        
        return nn.Sequential(*new_layers)
            

    if 'resnet' in args.arch or 'inception' in args.arch:
        in_size = model.fc.in_features
        model.fc = make_new_layers(in_size, embed_size, args.new_layers)
        return model.fc
        
    if 'densenet' in args.arch:
        in_size = model.classifier.in_features
        model.classifier = make_new_layers(in_size, embed_size, args.new_layers)
        return model.classifier
        
    if 'alexnet' in args.arch or 'vgg' in args.arch:
        in_size = model.classifier[-1].in_features
        model.classifier[-1] = make_new_layers(in_size, embed_size, args.new_layers)
        return model.classifier[-1]


def main(args):
    # Use CUDA?
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    # Load ImageNet
    print('Building ImageNet dataloader:', args.data)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    embed = None
    embeddings = None
    if args.embeddings:
        # Load Embeddings
        print('Loading label embeddings:', args.embeddings)
        embeddings = torch.from_numpy(np.load(args.embeddings))
        embed_size = embeddings.shape[1]
        embed = lambda idx: (idx, embeddings[idx])

    train_dataset = datasets.ImageFolder(traindir, transform=train_transform, target_transform=embed)
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform, target_transform=embed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # Create model
    model = models.__dict__[args.arch](pretrained=args.pretrained)
    print("Creating model: {} ({})".format(args.arch, 'pretrained' if args.pretrained else 'from scratch'))

    trained_params = model.parameters()
    # Change the last layer to embedding regression
    if args.embeddings:
        last_layer = adapt(model, embed_size, args)
        criterion = CosineDistance()
        if args.fix_weights:
            trained_params = last_layer.parameters()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()

    # Choose optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trained_params, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(trained_params, lr=args.lr,
                                    weight_decay=args.weight_decay)

    best_loss = None
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('WARNING: mismatching optimizer state, using a pristine optimizer..')
            if args.start_epoch < 0:
                args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        print('Evaluating on VAL set ...')
        args.log = sys.stdout
        evaluate(val_loader, model, criterion, args.start_epoch, embeddings, args)
        return

    # Start training
    if not args.run_dir:
        run_name = '{0[arch]}_b{0[batch_size]}_lr{0[lr]}_wd{0[weight_decay]}_optim{0[optimizer]}_nl{0[new_layers]}'.format(vars(args))
        run_name = 'pretrained_{}'.format(run_name) if args.pretrained else run_name
        run_name = 'fixed_{}'.format(run_name) if args.fix_weights else run_name
        run_name = 'cat2vec_{}'.format(run_name) if args.embeddings else run_name
        args.run_dir = os.path.join('runs', run_name)

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    print('Train parameters:', pformat(vars(args)))
    print('Training is starting...')
    log_file = os.path.join(args.run_dir, 'log.txt')
    args.log = open(log_file, 'a+')

    if not args.resume:
        print('Train parameters:', pformat(vars(args)), file=args.log)

    progress_bar = trange(args.start_epoch, args.epochs + 1, initial=args.start_epoch)
    for epoch in progress_bar:
        adjust_learning_rate(optimizer, epoch)

        # TRAIN
        progress_bar.set_description('TRAIN')
        train(train_loader, model, criterion, optimizer, epoch, args)
        # TEST
        progress_bar.set_description('VAL')
        val_loss = evaluate(val_loader, model, criterion, epoch, embeddings, args)

        is_best = best_loss is None or val_loss < best_loss
        best_loss = val_loss if best_loss is None else min(best_loss, val_loss)

        # SAVE MODEL
        fname = '{}_epoch_{:02d}.pth'.format(args.arch, epoch)
        fname = os.path.join(args.run_dir, fname)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)

    args.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat2Vec')
    parser.add_argument('data',
                        help='path to ImageNet dataset')
    parser.add_argument('--embeddings',
                        help='path to ImageNet label embeddings; only for cat2vec models')
    parser.add_argument('--run_dir', '-r', metavar='DIR',
                        help='where to save logs and snapshots')
    parser.add_argument('--arch', '-a', default='resnet18', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on custom restarts). Set to -1 to auto resume.')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('-o', '--optimizer', default='adam', metavar='OPTIM', choices=['sgd', 'adam'],
                        help='optimizer: sgd | adam (default: sgd)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('-l', '--log-interval', default=10, type=int, metavar='N',
                        help='train iterations between logging')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--fix-weights', default=False, action='store_true',
                        help='whether to fix the weights of the pre-trained part of the model')
    parser.add_argument('--new-layers', default=1, type=int, metavar='N',
                        help='new layers dedicated to embedding regression (default: 1, i.e. only the projection to the embeddings)')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='do not use CUDA acceleration')
    args = parser.parse_args()

    main(args)
