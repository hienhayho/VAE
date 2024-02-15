import torch
import argparse
from assets.utils import set_seed
from datasets.load_dataset import load_dataset
from models.load_model import load_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--model', type=str, default='originVAE', choices=['origin', 'cnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=["mnist", "cifar", "custom"], required=True)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=226)
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--latent_dim', type=int, default=200)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(226)
    print("Start loading dataset...")
    train_loader, test_loader = load_dataset(args)
    print("Finish loading dataset")
    model = load_model(args).to(DEVICE)
    print("------------------")
    
    print(model)
    
    print("------------------")
    model.train_epoch(train_loader, test_loader, args)
    

if __name__ == '__main__':
    main()