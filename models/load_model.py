def load_model(args):
    assert args.model, 'Model name should be given'
    
    if args.model == 'origin':
        from models import original_vae
        model = original_vae.originVAE(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model == 'cnn':
        pass
    else:
        raise NotImplementedError
    return model