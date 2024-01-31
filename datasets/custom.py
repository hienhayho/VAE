import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import os
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    """Custom dataset for training and testing VAE.

    Args:
        data_path (str): Path to the root directory of dataset
        transform (callable): Transform to be applied to the dataset
        train (bool): If True, load train dataset. If False, load test dataset
        
    Methods:
        load_image(image_path): Load image from image_path
           
    """
    def __init__(self, data_path, transform=None, train=True):
        super(CustomDataset, self).__init__()
        if train:
            self.data_path = os.path.join(data_path, 'train')
        else:
            self.data_path = os.path.join(data_path, 'test')
        self.data_path = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        self.transform = transform
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to tensor
        image = torch.from_numpy(image)
        # Add channel dimension
        image = image.unsqueeze(0)
        return image
        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image_path = self.data_path[idx]
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __str__(self):
        header = f'Dataset: Custom Dataset\n'
        path = f'Root directory: {self.data_path}\n'
        if self.train:
            subset = 'train'
        else:
            subset = 'test'
        num_samples = f'Num samples of {subset}: {len(self)}\n'
        return header + path + num_samples
    
def load_custom(args):
    assert args.dataset == 'custom', 'Dataset should be custom'
    assert args.data_path is not None, 'Dataset path should be given to load custom'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    train_dataset = CustomDataset(args.data_path, transform=transform, train=True)
    test_dataset = CustomDataset(args.data_path, transform=transform, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader