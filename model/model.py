from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


class Attention(nn.Module):
    def __init__(self, input_size: int, out: int = 24) -> None:
        super(Attention, self).__init__()
        self.input_size = input_size
        self.linear = nn.Sequential(
            nn.Linear(input_size, out),
            nn.ReLU(True),
            nn.Linear(out, 1)
        )

    def forward(self, encoder_outputs: torch.Tensor):
        bs = encoder_outputs.size(0)
        out = self.linear(encoder_outputs.view(-1, self.input_size))
        return F.softmax(out.view(bs, -1), dim=1).unsqueeze(2)


class ClassifierWithAttn(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder, vocab: Vocabulary, dropout: float) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = Attention(self.encoder.get_output_dim())
        self.linear = nn.Linear(self.encoder.get_output_dim(), 9)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_outputs = self.encoder(embeddings, mask)  # (batch_size, seq_len, hidden_size)
        attentions = self.attention(encoder_outputs)  # (batch_size, seq_len, 1)
        feats = (encoder_outputs * attentions).sum(dim=1)  # (batch_size, hidden_size)
        logits = self.linear(self.dropout(feats))  # (batch_size, 9)
        output = {"logits": logits, "attentions": attentions}

        if label is not None:
            loss = self.loss(logits, label.long())
            output["loss"] = loss

        output['encoder_outputs'] = encoder_outputs
        return output
