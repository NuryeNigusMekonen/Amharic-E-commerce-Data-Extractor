import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class TokenClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels, class_weights=None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.hidden2tag(self.dropout(outputs))

        # Fix CRF mask issue: Ensure first token is always on
        if attention_mask is not None and attention_mask.size(1) > 0:
           attention_mask[:, 0] = 1  #  Critical fix for CRF

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0  # Optional: Prevent -100 mismatch in CRF
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            return {"logits": emissions}


