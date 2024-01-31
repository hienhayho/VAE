from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_cifar(args):
    assert args.dataset == 'cifar', 'Dataset should be cifar'
    assert args.data_path is not None, 'Dataset path should be given to load cifar'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = CIFAR10(args.data_path, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(args.data_path, transform=transform, train=False, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader