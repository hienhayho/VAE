import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torch.nn import init
import cv2
import numpy as np
from torchsummary import summary
from assets import set_seed

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, train=False) -> None:
        super(Dataset).__init__()
        self.data_path = data_path
        self.transform = transform
        self.train = train
        if (self.train):
            self.data_path = os.path.join(self.data_path, 'train')
        else:
            self.data_path = os.path.join(self.data_path, 'test')
        self.image_paths = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def load_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        
        label = image_path.split('/')[-1].split('.')[0]
        assert label in ['cat', 'dog'], 'Label should be cat or dog'
        if label == 'cat':
            label = 0
        else:
            label = 1
        if self.transform:
            image = self.transform(image)
        return image, label
    
class Model(nn.Module): # 0.76
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.cnn5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512*8*8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.apply(self.initialize_weights)
    
    def forward(self, x):
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.relu(self.cnn3(x))
        x = self.relu(self.cnn4(x))
        x = self.relu(self.cnn5(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
    
    def forward(self, x):
        x = self.relu(self.cnn1(x))
        x = self.maxpool(x)
        x = self.relu(self.cnn2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG16(nn.Module): # (3, 224, 224)
    def __init__(self):
        super(VGG16, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.cnn7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.cnn9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.cnn10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.cnn12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.cnn13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn14 = nn.Conv2d(512, 4096, kernel_size=7)
        self.cnn15 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.cnn16 = nn.Conv2d(4096, 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.apply(self.initialize_weights)
        
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.maxpool1(x)
        x = self.relu(self.cnn3(x))
        x = self.relu(self.cnn4(x))
        x = self.maxpool2(x)
        x = self.relu(self.cnn5(x))
        x = self.relu(self.cnn6(x))
        x = self.relu(self.cnn7(x))
        x = self.maxpool3(x)
        x = self.relu(self.cnn8(x))
        x = self.relu(self.cnn9(x))
        x = self.relu(self.cnn10(x))
        x = self.maxpool4(x)
        x = self.relu(self.cnn11(x))
        x = self.relu(self.cnn12(x))
        x = self.relu(self.cnn13(x))
        x = self.maxpool5(x)
        x = self.relu(self.cnn14(x))
        x = self.relu(self.cnn15(x))
        x = self.cnn16(x)
        x = x.view(x.size(0), -1)
        return x

def main():
    data_path = '../data'
    batch_size = 128
    
    custom_tranform = transforms.Compose([
        transforms.Resize((32, 32)), 
        # transforms.RandomResizedCrop(size=(32, 32), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CustomDataset(data_path, transform=custom_tranform, train=True)
    test_dataset = CustomDataset(data_path, transform=custom_tranform, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 20
    model = Model().to('cuda')
    print(model)
    summary(model, (3, 32, 32))
    input("Press Enter to start training...")
    set_seed(226)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    global_test_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_correct += correct
        print(f'[Train] Epoch {epoch+1:03d}/{epochs:03d}:\tLoss: {total_loss:.4f}\tAccuracy: {train_correct/len(train_loader.dataset):.4f}')
        test_loss = 0
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to('cuda')
                labels = labels.to('cuda')
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                total_correct += correct
        test_accuracy = total_correct / len(test_loader.dataset)
        print(f'[Test]  Epoch {epoch+1:03d}/{epochs:03d}:\tLoss: {test_loss:.4f}\tAccuracy: {test_accuracy:.4f}')
        if test_accuracy > global_test_accuracy:
            global_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Save best model with accuracy: {test_accuracy:.4f}')
        print('-'*50)
    print(f"Best test accuracy: {global_test_accuracy:.4f}")

if __name__ == '__main__':
    main()