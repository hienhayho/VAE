from .mnist import load_mnist
from .cifar import load_cifar
from .custom import load_custom

def load_dataset(args):
    if args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(args)
    elif args.dataset == 'cifar':
        train_loader, test_loader = load_cifar(args)
    elif args.dataset == 'custom':
        pass
    else:
        raise NotImplementedError
    return train_loader, test_loader