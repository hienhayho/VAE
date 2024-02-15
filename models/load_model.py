def load_model(args):
    if args.model == 'origin':
        from models import originalVAE
        model = originalVAE.originVAE(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model == 'cnn':
        pass
    else:
        raise NotImplementedError
    return model