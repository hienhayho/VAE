import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.leakyrelu(self.linear1(x))
        x = self.leakyrelu(self.linear2(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leakyrelu(self.linear1(x))
        x = self.leakyrelu(self.linear2(x))
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
    
    
class OriginalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(OriginalVAE).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterization(self, mean, var):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD