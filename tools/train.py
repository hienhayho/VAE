import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--model', type=str, default='OriginalVAE', choices=['OriginalVAE', 'VAE_CNN'], required=True)
    parser.add_argument('--dataset', type=str, choices=["mnist", "catdog"], required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    return args

def load_model(args):
    if args.model == 'OriginalVAE':
        from models.OriginalVAE import Encoder, Decoder, OriginalVAE
        encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        decoder = Decoder(args.latent_dim, args.hidden_dim, args.output_dim)
        model = OriginalVAE(encoder, decoder)
    elif args.model == 'VAE_CNN':
        pass
    else:
        raise NotImplementedError
    return model

def load_data(args):
    if args.dataset == 'mnist':
        from datasets.mnist import load_mnist
        train_loader, test_loader = load_mnist(args.batch_size)
    elif args.dataset == 'catdog':
        from datasets.catdog import load_catdog
        train_loader, test_loader = load_catdog(args.batch_size)
    else:
        raise NotImplementedError
    return train_loader, test_loader