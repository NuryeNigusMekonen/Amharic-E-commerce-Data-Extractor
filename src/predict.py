import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.config import LABELS, MODEL_DIR


def load_model(model_path=MODEL_DIR):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def postprocess_bio_tags(predictions, words):
    """
    Fix BIO tag errors like:
    - I-TAG at beginning of sequence
    - I-TAG not following the same B-TAG or I-TAG
    """
    corrected = []
    prev_tag = "O"

    for word, tag in zip(words, predictions):
        if tag.startswith("I-"):
            base_tag = tag[2:]
            if prev_tag == f"B-{base_tag}" or prev_tag == f"I-{base_tag}":
                corrected.append(tag)
            else:
                corrected.append(f"B-{base_tag}")  # Correct to B- if bad I-
        else:
            corrected.append(tag)

        prev_tag = corrected[-1]

    return corrected


import torch
from transformers import AutoTokenizer
from src.config import LABELS, MODEL_LIST
from src.model import TokenClassificationModel


def load_model(model_name, model_dir):
    model = TokenClassificationModel(model_name=model_name, num_labels=len(LABELS))
    model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_text(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    words = text.strip().split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    word_ids = encoding.word_ids()
    inputs = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}

    with torch.no_grad():
        logits = model(**inputs)["logits"]
        mask = inputs["attention_mask"].bool()

        # Fix mask for CRF
        if mask.size(1) > 0:
            mask[:, 0] = True

        predictions = model.crf.decode(logits, mask=mask)

    final_labels = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        word = words[word_idx]
        label_id = predictions[0][idx]
        final_labels.append((word, LABELS[label_id]))
        prev_word_idx = word_idx

    return final_labels



