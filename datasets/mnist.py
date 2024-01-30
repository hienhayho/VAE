from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist(args):
    assert args.dataset == 'mnist', 'Dataset should be mnist'
    assert args.data_path is not None, 'Dataset path should be given'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = MNIST(args.data_path, transform=transform, train=True, download=True)
    test_dataset = MNIST(args.data_path, transform=transform, train=False, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader