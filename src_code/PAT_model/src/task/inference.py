import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
import pandas as pd
from torch.utils.data import DataLoader
import torch
import transformers
from utils.builder import get_model

class Predict:
    def __init__(self,config: Dict,answer_space):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.answer_space=answer_space
        self.model_name =config["model"]["name"]
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], config["model"]["name"], config["inference"]["checkpoint"], "pytorch_model.bin")
        self.test_path=config['inference']['test_dataset']
        self.bath_size=config['inference']['batch_size']
        self.use_id = config['data']['use_id']
        self.model = get_model(config,num_labels=len(self.answer_space))
    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
    
        
    # Load the model
        logging.info("Loading the {0} model...".format(self.model_name))
        
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test_set =self.get_dataloader(self.test_path)
        submits=[]
        ids=[]
        self.model.eval()
        with torch.no_grad():
            for item in test_set:
                output = self.model(item['id1_text'],item['id2_text'])
                preds = output["logits"].argmax(axis=-1).cpu().numpy()
                answers = [self.answer_space[i] for i in preds]
                submits.extend(answers)
                ids.extend(item['id'].cpu().numpy())

        data = {'id': ids,'label': submits }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)
        
    def load_test_set(self,file_path):
        test_set = pd.read_csv(file_path)
        annotations=[]
        if self.use_id:
            for i in range(len(test_set)):
                ann={
                    'id': test_set['id'][i],
                    'id1_text':str(test_set['id1'][i])+' '+test_set['id1_text'][i],
                    'id2_text':str(test_set['id2'][i])+' '+test_set['id2_text'][i]
                }
                annotations.append(ann)
        else:
            for i in range(len(test_set)):
                ann={
                    'id': test_set['id'][i],
                    'id1_text':test_set['id1_text'][i],
                    'id2_text':test_set['id2_text'][i]
                }
                annotations.append(ann)                
        return annotations

    def get_dataloader(self,file_path):
        dataset=self.load_test_set(file_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.bath_size,
            num_workers=2,
            shuffle=False,
        )
        return dataloader
