import torch
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score
from src.prepare_dataset import tokenize_and_align, get_dataset
from src.model import TokenClassificationModel
from src.config import LABELS, CLASS_WEIGHTS


from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain

from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain
import torch

from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain
import torch

def get_compute_metrics(crf, labels_list, device):
    from sklearn.metrics import precision_score, recall_score, f1_score

    def compute_metrics(p):
        emissions = torch.tensor(p.predictions, dtype=torch.float32).to(device)
        label_ids = torch.tensor(p.label_ids, dtype=torch.long).to(device)

        # Create a mask where label != -100
        mask = (label_ids != -100).bool()
        # FIX: ensure the first timestep mask is always ON for CRF
        if mask.size(1) > 0:
            mask[:, 0] = True
        # Replace -100 in labels with 0 for safe decoding
        label_ids = label_ids.clone()
        label_ids[label_ids == -100] = 0
        preds = crf.decode(emissions, mask)
        # Flatten predictions and labels
        true_labels = []
        true_preds = []
        for true, pred, m in zip(label_ids, preds, mask):
            seq_len = m.sum().item()
            true_labels.extend(true[:seq_len].tolist())
            true_preds.extend(pred[:seq_len])

        return {
            "precision": precision_score(true_labels, true_preds, average='macro', zero_division=0),
            "recall": recall_score(true_labels, true_preds, average='macro', zero_division=0),
            "f1": f1_score(true_labels, true_preds, average='macro', zero_division=0),
        }

    return compute_metrics

def train_model(model_name: str, model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw dataset (assumed to be Dataset or DatasetDict)
    raw_dataset = get_dataset()

    # Tokenize and align labels
    # Expecting tokenize_and_align returns (tokenized_dataset, tokenizer)
    tokenized_dataset, tokenizer = tokenize_and_align(raw_dataset, model_name)

    # Fix dataset splits:
    # Accept either:
    # - DatasetDict with splits 'train' and 'validation'
    # - DatasetDict with splits 'train' and 'test' -> rename 'test' to 'validation'
    # - Dataset with no splits -> split manually
    if isinstance(tokenized_dataset, DatasetDict):
        splits = list(tokenized_dataset.keys())
        if "train" in splits and "validation" in splits:
            train_dataset = tokenized_dataset["train"]
            eval_dataset = tokenized_dataset["validation"]
        elif "train" in splits and "test" in splits:
            # Rename 'test' split to 'validation' for Trainer API
            train_dataset = tokenized_dataset["train"]
            eval_dataset = tokenized_dataset["test"]
        else:
            raise ValueError(
                f"Unexpected dataset splits {splits}. Expected 'train'+'validation' or 'train'+'test'."
            )
    elif isinstance(tokenized_dataset, Dataset):
        # Split manually 80/20
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    else:
        raise ValueError("tokenized_dataset must be Dataset or DatasetDict")

    # Initialize model with CRF layer and class weights
    model = TokenClassificationModel(
        model_name=model_name,
        num_labels=len(LABELS),
        class_weights=torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    ).to(device)

    # Prepare training args
    args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{model_dir}/logs",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Get CRF layer and ensure it's on correct device
    crf = model.crf
    crf.to(device)

    compute_metrics = get_compute_metrics(crf, LABELS, device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    return trainer, model, tokenizer


# Example usage (make sure MODEL_LIST is defined in your main script)
if __name__ == "__main__":
    MODEL_LIST = {
        "your_model_name_here": "./saved_model_dir"
    }

    for model_name, model_dir in MODEL_LIST.items():
        print(f"\nTraining model: {model_name}")
        trainer, model, tokenizer = train_model(model_name, model_dir)
        print(f"Finished training {model_name}. Saved to {model_dir}")
