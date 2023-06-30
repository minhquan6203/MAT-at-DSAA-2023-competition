from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from encoder_module.init_encoder import build_encoder

class MutualAttentionTransformer(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(MutualAttentionTransformer, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.type_text_embedding=config["text_embedding"]['type']
        self.text_embbeding = build_text_embedding(config)          
        self.fusion = nn.Sequential(
            nn.Linear(self.intermediate_dims +self.intermediate_dims, self.intermediate_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = build_encoder(config)
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):

        if self.type_text_embedding=='pretrained':
            embbed_col1, mask_col1= self.text_embbeding(id1_text)
            embbed_col2, mask_col2 = self.text_embbeding(id2_text)
        else:
            embbed_col1 = self.text_embbeding(id1_text)
            embbed_col2 = self.text_embbeding(id2_text)
            mask_col1 = mask_col2 = None
            
        encoded_feature1, encoded_feature2 = self.encoder(embbed_col1, mask_col1, embbed_col2, mask_col2)
        
        feature1_attended = self.attention_weights(torch.tanh(encoded_feature1))
        feature2_attended = self.attention_weights(torch.tanh(encoded_feature2))
        
        attention_weights = torch.softmax(torch.cat([feature1_attended, feature2_attended], dim=1), dim=1)
        feature1_attended = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_feature1, dim=1)
        feature2_attended = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_feature2, dim=1)
        
        fused_output = self.fusion(torch.cat([feature1_attended, feature2_attended], dim=1))
        logits = self.classifier(fused_output)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

def createMutualAttentionTransformer(config: Dict, answer_space: List[str]) ->MutualAttentionTransformer:
    return MutualAttentionTransformer(config, num_labels=len(answer_space))