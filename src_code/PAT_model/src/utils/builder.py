from model.pat_model import createMutualAttentionTransformer,MutualAttentionTransformer
from data_utils.data_collator import createDataCollator

def build_model(config, answer_space):
    if config['model']['type_model']=='pat':
        return createMutualAttentionTransformer(config,answer_space)
    else:
        print('not support')
    
def get_model(config, num_labels):
    if config['model']['type_model']=='pat':
        return MutualAttentionTransformer(config,num_labels)
    else:
        print('not support')

def build_datacollator(config): 
    return createDataCollator(config)