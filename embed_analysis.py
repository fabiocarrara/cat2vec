import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.models as models

# Available PyTorch models for ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(args):
    label_embeds = None
    weights_embed = None

    if args.embeddings:
        print('Loading ebeddings: {}'.format(args.embeddings))
        dictionary_filename = args.embeddings.replace('.npy', '.txt')
        assert os.path.isfile(dictionary_filename), "Dictionary file not found: {}".format(dictionary_filename)
        print('Loading dictionary:', dictionary_filename)
        dictionary = np.array([line.rstrip() for line in open(dictionary_filename, 'rt')])
        label_embeds = torch.from_numpy(np.load(args.embeddings, mmap_mode='r'))

        title = os.path.splitext(os.path.basename(args.embeddings))[0]
        out_fname = 'embeds/{}_auto_similarity.pdf'.format(title)

        label_similarity = label_embeds @ label_embeds.t()
        plt.imshow(label_similarity)
        plt.title(title)
        plt.savefig(out_fname)
        print('Computed label similarity matrix: {}'.format(out_fname))
        plt.close()

    if args.arch:
        print('Using as embeddings the last layer of: {}'.format(args.arch))
        title = '{} last layer'.format(args.arch)
        model = models.__dict__[args.arch](pretrained=True)

        if 'resnet' in args.arch or 'inception' in args.arch:
            weights_embed = model.fc.weight.data
        elif 'densenet' in args.arch:
            weights_embed = model.classifier.weight.data
        elif 'alexnet' in args.arch or 'vgg' in args.arch:
            weights_embed = model.classifier[-1].weight.data

        weights_embed = F.normalize(weights_embed, p=2, dim=1)
        out_fname = 'embeds/{}_auto_similarity.pdf'.format(args.arch)
        weights_similarity = weights_embed @ weights_embed.t()
        plt.imshow(weights_similarity)
        plt.title(title)
        plt.savefig(out_fname)
        print('Computed weights similarity matrix: {}'.format(out_fname))
        plt.close()

    if args.embeddings and args.arch:
        differences = torch.triu(torch.abs(label_similarity - weights_similarity))
        differences = differences.numpy()
        positions = np.argsort(differences, axis=None)[::-1]
        positions = positions[:100]  # top-100 differences
        a_pos, b_pos = np.unravel_index(positions, (1000, 1000))

        data = pd.DataFrame(dict(a_labels=dictionary[a_pos],
                                 b_labels=dictionary[b_pos],
                                 label_sim=label_similarity[a_pos, b_pos],
                                 weight_sim=weights_similarity[a_pos, b_pos],
                                 distances=differences.ravel()[positions]))
        out_fname = 'embeds/top_differences_{}.csv'.format(args.arch)
        data.to_csv(out_fname)
        print('Saved top differences: {}'.format(out_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform analysis of the given category embeddings. '
                                                 'If -a is specified, the weights of the last classification layer '
                                                 'are taken as embeddings.')
    parser.add_argument('--embeddings', '-e', help='embeddings to analyze')
    parser.add_argument('--arch', '-a', default='resnet18', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    args = parser.parse_args()

    if not (args.embeddings or args.arch):
        parser.error('Add at least one between -a and -e.')

    main(args)
