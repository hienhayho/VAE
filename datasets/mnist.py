from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from assets.utils import set_seed

def load_mnist(args):
    set_seed(args.seed)
    assert args.dataset == 'mnist', 'Dataset should be mnist'
    assert args.data_path is not None, 'Dataset path should be given to load mnist'
    
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 

    train_dataset = MNIST(args.data_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(args.data_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
