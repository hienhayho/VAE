import torch
from assets import set_seed, parse_args
from datasets import load_dataset
from models import load_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Dataset preparation
    print("Start loading dataset...")
    train_loader, test_loader = load_dataset(args)
    print("Finish loading dataset")
    print("-"*80)
    
    # Model preparation
    model = load_model(args).to(DEVICE)
    print(model)
    print("-"*80)
    
    # Training
    model.train_epoch(train_loader, test_loader, args)
    
if __name__ == '__main__':
    main()