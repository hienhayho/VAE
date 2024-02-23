import torch
import cv2
import argparse
from torchvision.transforms import v2 as transforms
from test import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    return parser.parse_args()

def load_image(image_path: str, transform = None) -> torch.Tensor:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)
    return image

def main():
    args = parse_args()
    custom_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    model = Model()
    assert args.checkpoint_path, 'Checkpoint path is required'
    print("Loading model from checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint_path))
    image = load_image(args.image_path, custom_transform)
    print("-"*50)
    print("Inferencing...")
    output = model(image)
    output = torch.softmax(output, 1)
    value, predicted = torch.max(output, 1)
    print(f"Confidence: {value.item():.4f}")
    if predicted == 0:
        print('Cat')
    else:
        print('Dog')

if __name__ == '__main__':
    main()