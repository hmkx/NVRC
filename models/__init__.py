from . import hinerv


def create_model(args, logger, input):
    if args['type'] == 'HiNeRV':
        return hinerv.build_model(args['config'], logger, input)
    else:
        raise ValueError(f'Unknown model type: {args['type']}')