# -*-coding:utf-8 -*-

from collections import defaultdict


def decode_prediction(tokens, tags):
    """
    Decode NER Model Inference into {'PER': {'王一博','羽生结弦'}, 'ORG':{'证监会'}}
    1. if B & I has different ner type, will append [ERROR] token
    2. ## in word piece is removed
    """
    assert len(tokens) == len(tags), \
        'NER Decode {}!={}: token and pred_ids must have same len'.format(len(tokens), tags)
    result = defaultdict(set)
    entity = ''
    type1 = ''
    for token, tag in zip(tokens, tags):
        if 'B' in tag:
            if entity:
                result[type1].add(entity)
            entity = token.decode().replace('##', "") # to deal with wordpiece tokenizer
            type1 = tag.split('-')[1]
        elif 'I' in tag:
            type2 = tag.split('-')[1]
            if type1 == type2:
                entity += token.decode().replace('##', "")
            else:
                # Inconsistent BI pair
                entity += '[ERROR]'
        else:
            if entity:
                result[type1].add(entity)
            entity=''
    if entity:
        result[type1].add(entity)
    return result


def process_prediction(pred_dict, idx2tag):
    """
    Remove CLS, SEP, PAD from pred_ids, tokens and labels
    transform from id to tag for prediction
    """
    rm_tag = ['[CLS]', '[PAD]', '[SEP]']
    mask = [not (i.decode() in rm_tag) for i in pred_dict['tokens'] ]
    for key, val in pred_dict.items():
        pred_dict[key] = [i for i, j in zip(val, mask) if j ]
    # for corner case only
    mask = [not(idx2tag[i] in rm_tag)for i in pred_dict['pred_ids']]
    for key, val in pred_dict.items():
        pred_dict[key] = [i for i, j in zip(val, mask) if j ]

    pred_dict['sentence'] =  ''.join([i.decode() for i in pred_dict['tokens']])
    pred_dict['preds'] = [idx2tag[i] for i in pred_dict['pred_ids']] # encode for consistency
    pred_dict['labels'] = [idx2tag[i] for i in pred_dict['label_ids']]  # encode for consistency

    # add entity to dict
    pred_dict['label_entity'] = decode_prediction(pred_dict['tokens'], pred_dict['labels'])
    pred_dict['pred_entity'] = decode_prediction(pred_dict['tokens'], pred_dict['preds'])
    return pred_dict


if __name__ == '__main__':
    from data.msra.preprocess import TAG2IDX
    IDX2TAG = dict([(val, key )for key, val in TAG2IDX.items() ])

    inference = {
        'tokens': [i.encode() for i in '王一博,易烊千玺在北京天坛'],
        'preds': ['B-PER', 'I-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER',
                     'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
    }

    print(decode_prediction(inference['tokens'], inference['preds']))