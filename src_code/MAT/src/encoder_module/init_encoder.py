from encoder_module.multi_modal_encoder import CoAttentionEncoder

def build_encoder(config):
    if config['encoder']['type']=='co':
        return CoAttentionEncoder(config)
    else:
        print('not support the encoder you set')
