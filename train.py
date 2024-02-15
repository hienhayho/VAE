import torch
from assets.utils import set_seed
from assets.utils import parse_args
from datasets.load_dataset import load_dataset
from models.load_model import load_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Dataset preparation
    print("Start loading dataset...")
    train_loader, test_loader = load_dataset(args)
    print("Finish loading dataset")
    
    # Model preparation
    model = load_model(args).to(DEVICE)
    print("-"*80)
    print(model)
    print("-"*80)
    
    # Trainning
    model.train_epoch(train_loader, test_loader, args)
    
if __name__ == '__main__':
    main()