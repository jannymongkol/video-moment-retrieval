import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

def create_target_mask(start_frames, end_frames, num_frames, device):
    """
    Create binary target masks from start and end frame indices
    
    Args:
        start_frames: Tensor of start frame indices (batch_size,)
        end_frames: Tensor of end frame indices (batch_size,)
        num_frames: Number of frames in prediction
        device: Device to create tensor on
        
    Returns:
        Binary mask of shape (batch_size, num_frames)
    """
    batch_size = start_frames.size(0)
    masks = torch.zeros(batch_size, num_frames, device=device)
    
    for i in range(batch_size):
        start = start_frames[i]
        end = end_frames[i]
        
        # Create mask with 1s from start to end (inclusive)
        masks[i, start:end+1] = 1.0
        
    return masks

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        Dice Loss for binary classification with logits
        
        Args:
            smooth: Smoothing term to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, logits, intervals):
        """
        Calculate dice loss
        
        Args:
            logits: Model predictions as logits (batch_size, num_frames)
            start_frames: Start frame indices (batch_size,)
            end_frames: End frame indices (batch_size,)
            
        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        
        start_frames = [interval[0] for interval in intervals]
        end_frames = [interval[1] for interval in intervals]
        start_frames = torch.tensor(start_frames, device=logits.device)
        end_frames = torch.tensor(end_frames, device=logits.device)
        
        probs = torch.sigmoid(logits)
        
        # Create target masks
        num_frames = logits.size(1)
        targets = create_target_mask(start_frames, end_frames, num_frames, logits.device)
        
        # Calculate dice coefficient per sample
        batch_size = logits.size(0)
        dice_scores = torch.zeros(batch_size, device=logits.device)
        
        for i in range(batch_size):
            intersection = torch.sum(probs[i] * targets[i])
            union = torch.sum(probs[i]) + torch.sum(targets[i])
            dice_scores[i] = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Convert to loss (1 - dice)
        loss = 1.0 - dice_scores # (batch_size,)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class IntervalBCELoss(nn.Module):
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        Dice Loss for binary classification with logits
        
        Args:
            smooth: Smoothing term to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super(IntervalBCELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, intervals):
        """
        Calculate BCE loss
        
        Args:
            logits: Model predictions as logits (batch_size, num_frames)
            start_frames: Start frame indices (batch_size,)
            end_frames: End frame indices (batch_size,)
            
        Returns:
            BCE loss value
        
        """
        # Convert logits to probabilities
        
        start_frames = [interval[0] for interval in intervals]
        end_frames = [interval[1] for interval in intervals]
        start_frames = torch.tensor(start_frames, device=logits.device)
        end_frames = torch.tensor(end_frames, device=logits.device)
        
        probs = torch.sigmoid(logits)
        
        # Create target masks
        num_frames = logits.size(1)
        
        targets = create_target_mask(start_frames, end_frames, num_frames, logits.device)
        
        # Calculate dice coefficient per sample
        batch_size = logits.size(0)
        
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')  # (batch_size, num_frames)
        
        # Apply reduction
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:  # 'none'
            return bce_loss
        
class StartEndBCELoss(nn.Module):
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        Dice Loss for binary classification with logits
        
        Args:
            smooth: Smoothing term to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super(StartEndBCELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, intervals):
        """
        Calculate BCE loss
        
        Args:
            start_logits: Model predictions for start frame as logits (batch_size, num_frames)
            end_logits: Model predictions for end frame as logits (batch_size, num_frames)
            intervals: List of tuples containing start and end frame indices (batch_size, 2)
            
        Returns:
            BCE loss value
        
        """
        # Convert logits to probabilities
        
        start_logits, end_logits = logits
        
        start_frames = [interval[0] for interval in intervals]
        end_frames = [interval[1] for interval in intervals]
        
        probs_start = torch.sigmoid(start_logits)
        probs_end = torch.sigmoid(end_logits)
        
        ground_truth_start = torch.zeros_like(probs_start)
        ground_truth_end = torch.zeros_like(probs_start)
        for i in range(probs_start.shape[0]):
            ground_truth_start[i, start_frames[i]] = 1.0
            ground_truth_end[i, min(end_frames[i], probs_start.shape[1]-1)] = 1.0
        
        start_bce_loss = F.binary_cross_entropy(probs_start, ground_truth_start, reduction='none')  # (batch_size, num_frames)
        end_bce_loss = F.binary_cross_entropy(probs_end, ground_truth_end, reduction='none')  # (batch_size, num_frames)
        
        # Combine start and end BCE loss
        bce_loss = start_bce_loss + end_bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:  # 'none'
            return bce_loss

class MomentBERT(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=768, max_video_len=384, bert_trainable=False, prediction_head='in_out', num_hidden=0, inner_dim=512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = bert_trainable

        if num_hidden == 0:
            self.video_proj = nn.Linear(clip_dim, hidden_dim)
        else:
            layers = []
            start_size = clip_dim
            for _ in range(num_hidden):
                layers.append(nn.Linear(start_size, inner_dim))
                layers.append(nn.ReLU())
                start_size = inner_dim
            layers.append(nn.Linear(inner_dim, hidden_dim))
            
            self.video_proj = nn.Sequential(*layers)
        self.video_pos_embed = nn.Embedding(max_video_len, hidden_dim)

        if prediction_head == 'in_out':
            self.in_out_head = nn.Linear(hidden_dim, 1)
        
        if prediction_head == 'start_end':
            self.start_head = nn.Linear(hidden_dim, 1)
            self.end_head = nn.Linear(hidden_dim, 1)
        
        self.prediction_head = prediction_head
        self.max_video_len = max_video_len

    def forward(self, queries, video_clip_embeddings):
        """
        Forward pass through the model.
        This method takes the input queries and video embeddings, processes them through BERT,
        and runs the prediction head.
        
        Args:
            queries List(string): List of text queries.
            video_clip_embeddings (torch.tensor): Video embeddings of shape (N_frames, Clip_dim).
        Returns:
            if prediction_head == 'in_out':
                in_logits (torch.tensor): Logits for in-out frame prediction.
            if prediction_head == 'start_end':
                start_logits (torch.tensor): Logits for start frame prediction.
                end_logits (torch.tensor): Logits for end frame prediction.
        """
        B = len(queries)
        N_frames, clip_dim = video_clip_embeddings.shape
        video_clip_embeddings = video_clip_embeddings.unsqueeze(0).expand(B, N_frames, clip_dim)
        device = video_clip_embeddings.device

        # Use BERT tokenizer to process queries
        encodings = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].to(device) # (B, T)
        attention_mask = encodings['attention_mask'].to(device) # (B, T)
        
        text_embeds = self.bert.embeddings(input_ids=input_ids)  # (B, T, bert_hidden_dim)

        # Project video embeddings and add learned position encoding
        video_proj = self.video_proj(video_clip_embeddings)      # (B, N_frames, bert_hidden_dim)
        pos_ids = torch.arange(N_frames, device=device).unsqueeze(0).expand(B, N_frames)  # (B, N_frames)
        video_pos = self.video_pos_embed(pos_ids)                # (B, N_frames, bert_hidden_dim)
        video_embeds = video_proj + video_pos                    # (B, N_frames, bert_hidden_dim)
        
        # [SEP] token embedding
        sep_token_id = 102  # for BERT
        sep_token = self.bert.embeddings.word_embeddings(
            torch.tensor(sep_token_id, device=device)
        ).expand(B, 1, -1)  # (M, 1, hidden_dim)

        # Concatenate: [text tokens] + [SEP] + [video tokens] + [SEP]
        full_input = torch.cat([text_embeds, sep_token, video_embeds, sep_token], dim=1)  # (M, T + N_frames + 2, bert_hidden_dim)
        # Attention mask
        full_attention = torch.cat([
            attention_mask,                             # (M, T)
            torch.ones((B, 1), device=device),          # first SEP
            torch.ones((B, N_frames), device=device),          # video
            torch.ones((B, 1), device=device)           # final SEP
        ], dim=1)  # (B, T + N_frames + 2)
        
        # Token type ids
        token_type_ids = torch.cat([
            torch.zeros((B, input_ids.size(1)), device=device, dtype=torch.long),
            torch.zeros((B, 1), device=device, dtype=torch.long),
            torch.ones((B, N_frames), device=device, dtype=torch.long),
            torch.ones((B, 1), device=device, dtype=torch.long)
        ], dim=1, )  # (B, T + N_frames + 2)

        # Forward through frozen BERT
        outputs = self.bert(
            inputs_embeds=full_input,
            attention_mask=full_attention,
            token_type_ids=token_type_ids
        )
        
        contextual = outputs.last_hidden_state  # (B, T + N_frames + 2, hidden)
        video_contextual = contextual[:, input_ids.size(1) + 1 : -1, :]  # (B, N_frames, hidden)

        # Predict frame positions
        if self.prediction_head == 'start_end':
            start_logits = self.start_head(video_contextual).squeeze(-1)  # (B, N_frames)
            end_logits = self.end_head(video_contextual).squeeze(-1)      # (B, N_frames)

            return start_logits, end_logits

        elif self.prediction_head == 'in_out':
            in_logits = self.in_out_head(video_contextual).squeeze(-1)  # (B, N_frames)
            return in_logits


class MomentBERTv2(nn.Module):
    '''
    Same as Moment BERT, but uses BERT's positional embeddings instead of learned ones.
    Removes trailing [SEP] token.
    '''
    def __init__(
        self, 
        clip_dim=512, 
        hidden_dim=768, 
        max_video_len=384, 
        bert_trainable=False, 
        prediction_head='in_out', 
        num_hidden=0, 
        inner_dim=512
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = bert_trainable

        if num_hidden == 0:
            self.video_proj = nn.Linear(clip_dim, hidden_dim)
        else:
            layers = []
            start_size = clip_dim
            for _ in range(num_hidden):
                layers.append(nn.Linear(start_size, inner_dim))
                layers.append(nn.ReLU())
                start_size = inner_dim
            layers.append(nn.Linear(inner_dim, hidden_dim))
            self.video_proj = nn.Sequential(*layers)

        if prediction_head == 'in_out':
            self.in_out_head = nn.Linear(hidden_dim, 1)
        
        if prediction_head == 'start_end':
            self.start_head = nn.Linear(hidden_dim, 1)
            self.end_head = nn.Linear(hidden_dim, 1)
        
        self.prediction_head = prediction_head
        self.max_video_len = max_video_len

    def forward(self, queries, video_clip_embeddings):
        """
        Forward pass through the model.
        
        Args:
            queries (List[str]): List of text queries.
            video_clip_embeddings (torch.tensor): Video embeddings of shape (N_frames, Clip_dim).
        
        Returns:
            if prediction_head == 'in_out':
                in_logits (torch.tensor): Logits for in-out frame prediction.
            if prediction_head == 'start_end':
                start_logits (torch.tensor): Logits for start frame prediction.
                end_logits (torch.tensor): Logits for end frame prediction.
        """
        B = len(queries)
        N_frames, clip_dim = video_clip_embeddings.shape
        video_clip_embeddings = video_clip_embeddings.unsqueeze(0).expand(B, N_frames, clip_dim)
        device = video_clip_embeddings.device

        # Tokenize queries
        encodings = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(device)        # (B, T)
        attention_mask = encodings['attention_mask'].to(device)  # (B, T)
        
        text_embeds = self.bert.embeddings(input_ids=input_ids)  # (B, T, hidden_dim)
        
        T = text_embeds.shape[1]

        # Project and position-encode video embeddings using BERT's positional embeddings
        video_proj = self.video_proj(video_clip_embeddings)  # (B, N_frames, hidden_dim)
        video_position_ids = torch.arange(T+1, T+1+N_frames, device=device).unsqueeze(0).expand(B, N_frames)
        video_pos_embeds = self.bert.embeddings.position_embeddings(video_position_ids)
        video_embeds = video_proj + video_pos_embeds  # (B, N_frames, hidden_dim)

        # [SEP] token embedding
        sep_token_id = 102  # [SEP] token ID in BERT
        sep_token = self.bert.embeddings.word_embeddings(
            torch.tensor(sep_token_id, device=device)
        )
        sep_token += self.bert.embeddings.position_embeddings(
            torch.tensor(T, device=device)
        )
        sep_token = sep_token.expand(B, 1, -1)  # (B, 1, hidden_dim)

        # Concatenate: [text] + [SEP] + [video]
        full_input = torch.cat([text_embeds, sep_token, video_embeds], dim=1)  # (B, T + 1 + N_frames, hidden_dim)

        # Attention mask
        full_attention = torch.cat([
            attention_mask,                           # (B, T)
            torch.ones((B, 1), device=device),        # SEP
            torch.ones((B, N_frames), device=device), # video
        ], dim=1)

        # Token type IDs
        token_type_ids = torch.cat([
            torch.zeros((B, input_ids.size(1)), device=device, dtype=torch.long),  # text
            torch.zeros((B, 1), device=device, dtype=torch.long),                  # SEP
            torch.ones((B, N_frames), device=device, dtype=torch.long),           # video
        ], dim=1)

        # Forward through BERT
        outputs = self.bert(
            inputs_embeds=full_input,
            attention_mask=full_attention,
            token_type_ids=token_type_ids
        )
        
        contextual = outputs.last_hidden_state  # (B, T + 1 + N_frames, hidden_dim)
        video_contextual = contextual[:, input_ids.size(1) + 1 :, :]  # (B, N_frames, hidden_dim)

        if self.prediction_head == 'start_end':
            start_logits = self.start_head(video_contextual).squeeze(-1)  # (B, N_frames)
            end_logits = self.end_head(video_contextual).squeeze(-1)      # (B, N_frames)
            return start_logits, end_logits

        elif self.prediction_head == 'in_out':
            in_logits = self.in_out_head(video_contextual).squeeze(-1)  # (B, N_frames)
            return in_logits