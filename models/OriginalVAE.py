import torch
import torch.nn as nn
import os
import numpy as np
from assets.logging import configure_logging, log_info, initial_logging
from assets.date import get_current_date
import time

class Encoder(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 200):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
    
class OriginalVAE(nn.Module):
    def __init__(self, encoder, decoder, input_dim = 784, hidden_dim = 400, latent_dim = 200):
        super(OriginalVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.Encoder = encoder(input_dim, hidden_dim, latent_dim)
        self.Decoder = decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterization(self, mean, var):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var
    

    def train_epoch(self, train_loader, test_loader, args):
        assert args.save_path is not None, 'Save path should be given to train model'
        
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
        
        initial_logging('OriginalVAE', args, subfolder=current_day)
        
        new_save_path = os.path.join(args.save_path, current_day)
        
        # Load model to device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(DEVICE)
        
        # Define optimizer
        if args.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        elif args.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)
        else:
            raise NotImplementedError
        
        # Start training, count time elapsed
        start_time = time.time()
        log_info("Start training...")
        min_loss = np.inf
        
        for epoch in range(args.epochs):
            self.train()
            train_loss = 0
            train_start_time = time.time()
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(args.batch_size, self.input_dim)
                data = data.to(DEVICE)
                optimizer.zero_grad()
                x_hat, mean, log_var = self(data)
                loss = loss_function(data, x_hat, mean, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_end_time = time.time()
            train_loss /= (batch_idx*args.batch_size)
            info = "[Train] Epoch " + str(epoch + 1) + ":" + "\tAverage Loss: " + str(train_loss) + "\tEpoch training time: {:.2f} seconds".format(train_end_time - train_start_time)
            log_info(info)
            
            if epoch % args.val_interval == 0:
                self.eval()
                test_loss = 0
                with torch.no_grad():
                    for batch_idx, (data, _) in enumerate(test_loader):
                        data = data.view(args.batch_size, self.input_dim)
                        data = data.to(DEVICE)
                        x_hat, mean, log_var = self(data)
                        loss = loss_function(data, x_hat, mean, log_var)
                        test_loss += loss.item()
                test_loss /= (batch_idx*args.batch_size)
                info = "[Val] Epoch " + str(epoch + 1) + ":" + "\tAverage Loss: " + str(test_loss)
                log_info(info)
                
                if test_loss < min_loss and args.save_model:
                    for f in os.listdir(new_save_path):
                        if f.endswith('.pth'):
                            os.remove(os.path.join(new_save_path, f))
                            log_info("Remove {}".format(f))
                    
                    min_loss = test_loss
                    log_info("Model with lowest loss {:.4f} is saved at {}".format(min_loss, new_save_path))
                    torch.save(self.state_dict(), os.path.join(new_save_path, f'Epoch_{epoch + 1}_{test_loss:4f}.pth'))
        end_time = time.time()
        log_info("Finish training")
        log_info("Time elapsed: {:.2f} seconds".format(end_time - start_time))
        
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD