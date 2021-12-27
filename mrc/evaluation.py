"""
    Evaluation
    之前把数据都pass through model直接写pred的方式和线上会有出入，这里还是做了区分所以重写了evaluation

"""

import pandas as pd
def bio_extract_entity(bio_ids, tokens, tag2idx):
    entity_list = []
    entity = []
    for i ,j in zip(bio_ids, tokens):
        if i == tag2idx['B']:
            if entity:
                entity_list.append(''.join(entity))
            entity = [j]
        elif i == tag2idx['I']:
            entity+=[j]
        else:
            if entity:
                entity_list.append(''.join(entity))
                entity = []
    if entity:
        entity_list.append(''.join(entity))
    return entity_list


