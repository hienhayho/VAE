def load_model(args):
    if args.model == 'origin':
        from .OriginalVAE import Encoder, Decoder, OriginalVAE
        model = OriginalVAE(encoder=Encoder, decoder=Decoder)
    elif args.model == 'cnn':
        pass
    else:
        raise NotImplementedError
    return model