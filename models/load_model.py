def load_model(args):
    if args.model == 'origin':
        from .originalVAE import Encoder, Decoder, originVAE
        model = originVAE(encoder=Encoder, decoder=Decoder)
    elif args.model == 'cnn':
        pass
    else:
        raise NotImplementedError
    return model