import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models.load_model import load_model
from datasets.load_dataset import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    # Set seed for random module
    torch.manual_seed(seed)
    # Set seed for numpy module
    np.random.seed(seed)
    # Set seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Sử dụng seed cố định (ví dụ: seed = 42)
set_seed(42)

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = load_model(args)
    print("------------------")
    print(model)
    print("------------------")
    print("Start loading dataset...")
    train_loader, test_loader = load_dataset(args)
    print("Finish loading dataset")
    print("------------------")

    # dataset_path = './'

    
    # print(DEVICE)

    # batch_size = 100

    # x_dim  = 784
    # hidden_dim = 400
    # latent_dim = 200

    # lr = 1e-3

    # epochs = 30

    # mnist_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # kwargs = {'num_workers': 1, 'pin_memory': True} 

    # train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    # test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    # # Train
    
    # encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    # decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    # model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    model.train_epoch(train_loader, test_loader, args)
    # def loss_function(x, x_hat, mean, log_var):
    #     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    #     KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    #     return reproduction_loss + KLD


    # optimizer = Adam(model.parameters(), lr=lr)
    # print("Start training VAE...")
    # model.train()

    # for epoch in range(epochs):
    #     overall_loss = 0
    #     for batch_idx, (x, _) in enumerate(train_loader):
    #         x = x.view(batch_size, x_dim)
    #         x = x.to(DEVICE)

    #         optimizer.zero_grad()

    #         x_hat, mean, log_var = model(x)
    #         loss = loss_function(x, x_hat, mean, log_var)
            
    #         overall_loss += loss.item()
            
    #         loss.backward()
    #         optimizer.step()
            
    #     print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    # print("Finish!!")
    

if __name__ == '__main__':
    main()