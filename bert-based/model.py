import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

class MomentBERT(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=768, max_video_len=256):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        self.video_proj = nn.Linear(clip_dim, hidden_dim)
        self.video_pos_embed = nn.Embedding(max_video_len, hidden_dim)

        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, video_clip_embeddings):
        """
        input_ids: (M, T) — tokenized text queries
        attention_mask: (M, T)
        video_clip_embeddings: (M, N, clip_dim) — one per query
            where M is total number of (video, query) pairs
        """

        M, N, _ = video_clip_embeddings.shape
        device = video_clip_embeddings.device

        # Text embeddings
        text_embeds = self.bert.embeddings(input_ids=input_ids)  # (M, T, hidden_dim)

        # Project video embeddings and add learned position encoding
        video_proj = self.video_proj(video_clip_embeddings)      # (M, N, hidden_dim)
        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(M, N)  # (M, N)
        video_pos = self.video_pos_embed(pos_ids)                # (M, N, hidden_dim)
        video_embeds = video_proj + video_pos                    # (M, N, hidden_dim)

        # [SEP] token embedding
        sep_token_id = 102  # for BERT
        sep_token = self.bert.embeddings.word_embeddings(
            torch.tensor(sep_token_id, device=device)
        ).expand(M, 1, -1)  # (M, 1, hidden_dim)

        # Concatenate: [text tokens] + [SEP] + [video tokens] + [SEP]
        full_input = torch.cat([text_embeds, sep_token, video_embeds, sep_token], dim=1)  # (M, T + N + 2, hidden)

        # Attention mask
        full_attention = torch.cat([
            attention_mask,                             # (M, T)
            torch.ones((M, 1), device=device),          # first SEP
            torch.ones((M, N), device=device),          # video
            torch.ones((M, 1), device=device)           # final SEP
        ], dim=1)  # (M, T + N + 2)

        # Token type ids
        token_type_ids = torch.cat([
            torch.zeros((M, input_ids.size(1)), device=device),  # text
            torch.zeros((M, 1), device=device),
            torch.ones((M, N), device=device),                   # video
            torch.ones((M, 1), device=device)
        ], dim=1)  # (M, T + N + 2)

        # Forward through frozen BERT
        outputs = self.bert(
            inputs_embeds=full_input,
            attention_mask=full_attention,
            token_type_ids=token_type_ids
        )

        contextual = outputs.last_hidden_state  # (M, T + N + 2, hidden)
        video_contextual = contextual[:, input_ids.size(1) + 1 : -1, :]  # (M, N, hidden)

        # Predict frame positions
        start_logits = self.start_head(video_contextual).squeeze(-1)  # (M, N)
        end_logits = self.end_head(video_contextual).squeeze(-1)      # (M, N)

        return start_logits, end_logits
