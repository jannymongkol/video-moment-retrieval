import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import os
import sys
from torch.utils.data import DataLoader

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frame_by_frame_bert.dataloader import TemporalGroundingDataset, collate_fn

def build_mlp(input_dim, output_dim, num_layers, hidden_dim=512, activation=nn.ReLU, use_layernorm=True):
    layers = []
    current_dim = input_dim

    for _ in range(num_layers - 1):
        next_dim = hidden_dim
        layers.append(nn.Linear(current_dim, next_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(next_dim))
        layers.append(activation())
        
        current_dim = next_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

class TemporalGroundingModel(nn.Module):
    def __init__(self, clip_dim=512, bert_model_name="bert-base-uncased", projection_dim=768, frozen_bert=True):
        super(TemporalGroundingModel, self).__init__()

        # BERT Model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = not frozen_bert
        
        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Frame Embedding Projection (CLIP space -> BERT token space)
        self.frame_projection = build_mlp(
            input_dim=clip_dim,
            output_dim=projection_dim,
            num_layers=2,
            hidden_dim=512,
            activation=nn.GELU,
            use_layernorm=True
        )

        # Classification head (output for frame relevance)
        self.classification_head = build_mlp(
            input_dim=projection_dim,
            output_dim=1,
            num_layers=1
        )

    def forward(self, clip_embeddings, texts):
        
        tokenized = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(clip_embeddings.device)
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        batch_size = input_ids.shape[0]
        max_text_length = input_ids.shape[1]
        
        # 1. Project Frame Embeddings from CLIP space to BERT token space
        frame_embeddings = self.frame_projection(clip_embeddings)

        # 2. Add Positional Encoding
        pos_encoding = self.bert.embeddings.position_embeddings(
            torch.tensor([max_text_length], device=clip_embeddings.device))
                
        frame_embeddings += pos_encoding
        
        # Concatenate text embeddings and frame embeddings (Text to Frame)
        # Text embeddings from BERT + Frame Embedding
        text_embeddings = self.bert.embeddings(input_ids=input_ids)
                
        full_input_embeddings = torch.cat([text_embeddings, frame_embeddings.unsqueeze(1)], dim=1)
        
        # 3. Create attention mask
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=1)

        # 4. Pass through BERT
        outputs = self.bert(inputs_embeds=full_input_embeddings, attention_mask=attention_mask)

        # 5. Use CLS token output for classification
        cls_output = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)
        
        # 6. Classify the frame as relevant (in interval) or not
        logits = self.classification_head(cls_output)
        
        return logits