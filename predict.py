import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from glob2 import glob as glob
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# Available PyTorch models for ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class ImageListDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = default_loader(path)

        if self.transform:
            img = self.transform(img)

        return img


def predict(data, model, args):
    model.eval()

    if args.cuda:
        model.cuda()

    predictions = []
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        X = sample_batched
        if args.cuda:
            X = X.cuda()
        X = torch.autograd.Variable(X, volatile=True)

        Y_hat = model(X)
        Y_hat = F.normalize(Y_hat.data, p=2, dim=1)
        predictions.append(Y_hat.cpu())
        
    return torch.cat(predictions, dim=0)


def find_nearest_embeddings(predictions, embeddings, k=5):
#    if args.cuda:
#        embeddings = embeddings.cuda()
#        predictions = predictions.cuda()

    # Embeddings should be already L2-normalized -- if not
    # uncomment the following line.
    # embeddings = F.normalize(embeddings, p=2, dim=1)

    scores = predictions.mm(embeddings.t())
    S, I = scores.topk(k, dim=1, largest=True, sorted=True)

    return S, I


def compare_predictions(S, I, S2, I2):
    pass
    # for path, score, idx in zip(urls, S, I):
    #     print(path)
    #     print('-----')
    #     for s, i in zip(score, idx):
    #         print('{}: {} ({})'.format(s, dictionary[i], i))
    #         # predictions.append([path, s, dictionary[i], i])

    # results = pd.DataFrame().from_records(predictions)
    # results.to_csv('predictions.csv')


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

    # Load Embeddings
    dictionary_filename = args.embeddings.replace('.npy', '.txt')
    assert os.path.isfile(dictionary_filename), "Dictionary file not found: {}".format(dictionary_filename)
    print('Loading dictionary:', dictionary_filename)
    dictionary = np.array([line.rstrip() for line in open(dictionary_filename, 'rt')])

    print('Loading label embeddings:', args.embeddings)
    embeddings = torch.from_numpy(np.load(args.embeddings))
    embed_size = embeddings.shape[1]
    embed = lambda idx: (idx, embeddings[idx])

    # Load images (flatten folder)

    def find_images(path):
        images = []
        for ext in ('jpg', 'jpeg', 'gif', 'png'):
            pattern = os.path.join(path, '**/*.{}'.format(ext))
            images.extend(glob(pattern))
        return images

    urls = [find_images(i) if os.path.isdir(i) else (i,) for i in args.data]
    urls = [item for list in urls for item in list]  # flatten out dirs

    print('Image found: {}'.format(len(urls)))
    dataset = ImageListDataset(urls, transform=val_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers)
    # Create model
    model = models.__dict__[args.arch]()
    print("Creating model: {} ({})".format(args.arch, args.checkpoint))

    # Change the last layer to embedding regression
    adapt(model, embed_size, args)
    assert os.path.isfile(args.checkpoint), "Checkpoint file not found: {}".format(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    predicted_embeds = predict(loader, model, args)
    

    if args.output:
        out_fname = args.output if args.output.endswith('.npy') else '{}.npy'.format(args.output)
        np.save(out_fname, predicted_embeds.numpy())

    del model
    torch.cuda.empty_cache()

    # DEBUG: also print ILSVRC12 nearest embedding
    i_dictionary = np.array([line.rstrip() for line in open('ilsvrc12_cat_embeds_l2norm.txt')])
    i_embeddings = torch.from_numpy(np.load('ilsvrc12_cat_embeds_l2norm.npy'))

    top_scores, top_indexes = find_nearest_embeddings(predicted_embeds, embeddings, k=5)
    i_top_scores, i_top_indexes = find_nearest_embeddings(predicted_embeds, i_embeddings, k=5)

    top1_embeds = embeddings[top_indexes[:, 0]]
    i_top1_embeds = i_embeddings[i_top_indexes[:, 0]]
    
    top1_labels = dictionary[top_indexes[:, 0]]
    i_top1_labels = i_dictionary[i_top_indexes[:, 0]]

    top1_labels_similarity = torch.sum(top1_embeds * i_top1_embeds, dim=1)

    results = {
        'URL': urls,
        'GoogleNewsScore': top_scores[:, 0],
        'GoogleNewsPrediction': top1_labels,
        'GoogleNewsEmbIdx': top_indexes[:, 0],
        'ILSVRC12Score': i_top_scores[:, 0],
        'ILSVRC12Prediction': i_top1_labels,
        'ILSVRC12EmbIdx': i_top_indexes[:, 0],
        'LabelCosineSimilarity': top1_labels_similarity
    }

    results = pd.DataFrame(results)
    if args.output:
        out_fname = args.output if args.output.endswith('.csv') else '{}.csv'.format(args.output)
        results.to_csv(out_fname, index=False)
    else:
        print (results)

    # for path, score, idx, i_score, i_idx in zip(urls, top_scores, top_indexes, i_top_scores, i_top_indexes):
    #     print(path)
    #     print('-----')
    #     for s, i, si, ii in zip(score, idx, i_score, i_idx):  # iterate over top-K scores
    #         print('{}: {} ({}) \t {}: {} ({})'.format(s, dictionary[i], i, si, i_dictionary[ii], ii))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat2Vec Prediction')

    parser.add_argument('data', nargs='+',
                        help='input images')
    parser.add_argument('-o', '--output',
                        help='where to store predicted embeddings')
    parser.add_argument('-e', '--embeddings',
                        help='path to target embeddings; only for label prediction with cosine similarity')
    parser.add_argument('-c', '--checkpoint',
                        help='path to model checkpoint; only for cat2vec models')
    parser.add_argument('--arch', '-a', default='resnet18', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='do not use CUDA acceleration')
    args = parser.parse_args()

    main(args)
