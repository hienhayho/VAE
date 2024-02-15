import torch
from assets.utils import set_seed
from assets.utils import parse_args
from datasets.load_dataset import load_dataset
from models.load_model import load_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    set_seed(226)
    print("Start loading dataset...")
    train_loader, test_loader = load_dataset(args)
    print("Finish loading dataset")
    model = load_model(args).to(DEVICE)
    print("------------------")
    print(model)
    print("------------------")
    model.train_epoch(train_loader, test_loader, args)
    

if __name__ == '__main__':
    main()