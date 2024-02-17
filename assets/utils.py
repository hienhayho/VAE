import random
import numpy as np
import torch
import argparse
import os
from assets.logging import configure_logging, initial_logging
from assets.date import get_current_date

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--model', type=str, default='originVAE', choices=['origin', 'cnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=["mnist", "cifar", "custom"], required=True)
    parser.add_argument('--data_path', type=str, required=True)
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

def init_result_path(args):
    # Create save path folder if not exist        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, 'training.log')

    # Make folder train name with current day
    current_day = get_current_date()
    os.makedirs(os.path.join(args.save_path, current_day))
    log_file_path = os.path.join(args.save_path, current_day, 'training.log')

    if not os.path.exists(log_file_path):
        os.system('touch {}'.format(log_file_path))
        # If the log file doesn't exist, create it and configure logging
        configure_logging(log_file_path)

    if args.model == 'origin':
        model_name = 'originVAE'
    elif args.model == 'cnn':
        model_name = 'cnnVAE'
    else:
        raise NotImplementedError
    
    initial_logging(model_name, args, subfolder=current_day)
    new_save_path = os.path.join(args.save_path, current_day)
    
    return new_save_path