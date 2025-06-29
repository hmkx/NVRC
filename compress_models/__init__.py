from . import nvrc


def create_compress_model(logger, model, cfgs):
    if cfgs['type'] == 'NVRC':
        return nvrc.build_compress_model(logger, model, cfgs['config'])
    else:
        raise ValueError(f'Unknown compress model type: {cfgs['type']}')