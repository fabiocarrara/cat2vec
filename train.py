import argparse
import os
import shutil
from collections import OrderedDict

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm, trange

# Available PyTorch models for ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(data, model, criterion, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X, (iY, Y) = sample_batched
        if args.cuda:
            X = X.cuda()
            Y = Y.cuda(async=True)
        X = torch.autograd.Variable(X, requires_grad=False)
        Y = torch.autograd.Variable(Y, requires_grad=False)

        # forward and backward pass
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = criterion(Y, Y_hat).mean()
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
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        tmp = correct_k.mul_(100.0 / batch_size)
        res.append(tmp.data[0])
    return res


def evaluate(data, model, criterion, epoch, embeddings, args):
    model.eval()

    if args.cuda:
        embeddings = embeddings.cuda()
    embeddings = torch.autograd.Variable(embeddings, requires_grad=False)

    avg_acc1 = 0.0
    avg_acc5 = 0.0
    avg_loss = 0.0
    n_samples = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X, (iY, Y) = sample_batched
        if args.cuda:
            X = X.cuda()
            Y = Y.cuda(async=True)
            iY = iY.cuda(async=True)
        X = torch.autograd.Variable(X, volatile=True)
        Y = torch.autograd.Variable(Y, volatile=True)
        iY = torch.autograd.Variable(iY, volatile=True)

        Y_hat = model(X)
        pred = Y_hat @ embeddings.t()
        acc1, acc5 = accuracy(pred, iY, topk=(1, 5))

        loss = criterion(Y, Y_hat).mean()
        loss = loss.data[0]

        avg_loss += loss
        avg_acc1 += acc1
        avg_acc5 += acc5
        n_samples += len(Y)

        progress_bar.set_postfix(OrderedDict(
            loss='{:.2}'.format(loss),
            acc1='{:.2}%'.format(acc1),
            acc5='{:.2}%'.format(acc5)
        ))

    avg_loss /= n_samples
    avg_acc1 /= n_samples
    avg_acc5 /= n_samples

    print('Test Epoch {}: Loss = {:.5} Acc@1 = {:.3}% Acc@5 = {:.3}%'.format(epoch, avg_loss, avg_acc1, avg_acc5))

    return avg_loss


def adapt(model, embed_size, args):
    if 'resnet' in args.arch or 'inception' in args.arch:
        in_size = model.fc.in_features
        model.fc = torch.nn.Linear(in_size, embed_size)
    elif 'densenet' in args.arch:
        in_size = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_size, embed_size)
    elif 'alexnet' in args.arch or 'vgg' in args.arch:
        in_size = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_size, embed_size)


def main(args):
    # Use CUDA?
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    # Load Embeddings
    print('Loading label embeddings:', args.embeddings)
    embeddings = torch.from_numpy(np.load(args.embeddings))
    embed_size = embeddings.shape[1]

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

    # Change the last layer to embedding regression
    adapt(model, embed_size, args)
    criterion = torch.nn.CosineSimilarity()

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = None
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

            # Start training
    print('Training ({} epochs) is starting...'.format(args.epochs))
    progress_bar = trange(args.start_epoch, args.epochs + 1)
    for epoch in progress_bar:
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
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat2Vec')
    parser.add_argument('data',
                        help='path to ImageNet dataset')
    parser.add_argument('embeddings',
                        help='path to ImageNet label embeddings')
    parser.add_argument('--arch', '-a', default='resnet18', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=300, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('-l', '--log-interval', default=10, type=int, metavar='N',
                        help='train iterations between logging')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='do not use CUDA acceleration')
    args = parser.parse_args()

    main(args)
