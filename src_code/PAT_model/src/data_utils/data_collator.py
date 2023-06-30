import os
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import torch

@dataclass
class data_Collator:
    def __init__(self, config: Dict):
        self.use_id = config['data']['use_id']
    def __call__(self, raw_batch_dict):
        if self.use_id:
            return {
                'id1_text':[str(ann["id1"])+' '+ann["id1_text"] for ann in raw_batch_dict],
                'id2_text': [str(ann["id2"])+' '+ann["id2_text"] for ann in raw_batch_dict],
                'labels': torch.tensor([ann["label"] for ann in raw_batch_dict],
                dtype=torch.int64
                ),
            }
        else:
            return {
                'id1_text':[ann["id1_text"] for ann in raw_batch_dict],
                'id2_text': [ann["id2_text"] for ann in raw_batch_dict],
                'labels': torch.tensor([ann["label"] for ann in raw_batch_dict],
                dtype=torch.int64
                ),
            }

def createDataCollator(config: Dict) -> data_Collator:
    collator = data_Collator(config)
    return collator
