import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm

# Available PyTorch models for ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


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


def evaluate_cosine(data, model, embeddings, args):
    criterion = torch.nn.CosineSimilarity()

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()
        embeddings = embeddings.cuda()

    embeddings = torch.autograd.Variable(embeddings, requires_grad=False)

    model.eval()

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
        pred = Y_hat.mm(embeddings.t())

        acc1, acc5 = accuracy(pred, iY, topk=(1, 5))

        loss = - criterion(Y, Y_hat).mean()
        loss = loss.data[0]

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

    print('Loss = {:6.4f} Acc@1 = {:6.3f}% Acc@5 = {:6.3f}%'.format(avg_loss, avg_acc1, avg_acc5))

    return avg_loss


def evaluate(data, model, args):
    criterion = torch.nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()

    model.eval()

    avg_acc1 = 0.0
    avg_acc5 = 0.0
    avg_loss = 0.0
    n_samples = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X, Y = sample_batched
        if args.cuda:
            X = X.cuda()
            Y = Y.cuda(async=True)
        X = torch.autograd.Variable(X, volatile=True)
        Y = torch.autograd.Variable(Y, volatile=True)

        Y_hat = model(X)
        acc1, acc5 = accuracy(Y_hat, Y, topk=(1, 5))

        loss = criterion(Y_hat, Y).mean()
        loss = loss.data[0]

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

    print('Loss = {:6.4f} Acc@1 = {:6.3f}% Acc@5 = {:6.3f}%'.format(avg_loss, avg_acc1, avg_acc5))

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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    embed = None
    if args.embeddings:
        # Load Embeddings
        print('Loading label embeddings:', args.embeddings)
        embeddings = torch.from_numpy(np.load(args.embeddings))
        embed_size = embeddings.shape[1]
        embed = lambda idx: (idx, embeddings[idx])

    # Load folder (e.g: ImageNet)
    print('Building dataloader:', args.data)
    dataset = datasets.ImageFolder(args.data, transform=val_transform, target_transform=embed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers)
    # Create model
    pretrained = args.embeddings is None
    model = models.__dict__[args.arch](pretrained=pretrained)
    snapshot = 'pretrained' if pretrained else args.checkpoint if args.checkpoint else 'from-scratch'
    print("Creating model: {} ({})".format(args.arch, snapshot))

    if args.embeddings:
        # Change the last layer to embedding regression
        adapt(model, embed_size, args)
        assert os.path.isfile(args.checkpoint), "Checkpoint file not found: {}".format(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        evaluate_cosine(loader, model, embeddings, args)
    else:
        evaluate(loader, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat2Vec Prediction')

    parser.add_argument('data',
                        help='path to folder with images')
    parser.add_argument('-c', '--checkpoint',
                        help='path to model checkpoint; only for cat2vec models')
    parser.add_argument('-e', '--embeddings',
                        help='path to ImageNet label embeddings; only for vat2vec models')
    parser.add_argument('--arch', '-a', default='resnet18', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers if given data is a folder (default: 4)')
    parser.add_argument('-b', '--batch-size', default=300, type=int, metavar='N',
                        help='mini-batch size (default: 300)')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='do not use CUDA acceleration')
    args = parser.parse_args()

    main(args)
