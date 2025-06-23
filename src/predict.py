import torch
from transformers import AutoTokenizer
from src.model import TokenClassificationModel
from src.config import LABELS
from safetensors.torch import load_file
import inspect


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
            if prev_tag in [f"B-{base_tag}", f"I-{base_tag}"]:
                corrected.append(tag)
            else:
                corrected.append(f"B-{base_tag}")
        else:
            corrected.append(tag)
        prev_tag = corrected[-1]

    return corrected


def load_model(model_name, checkpoint_dir):
    """
    Loads CRF-based model from HuggingFace Trainer checkpoint (safetensors format).
    """
    model = TokenClassificationModel(model_name=model_name, num_labels=len(LABELS))
    
    # Load safely from safetensors
    model.load_state_dict(load_file(f"{checkpoint_dir}/model.safetensors"), strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.eval()
    return model, tokenizer


def predict_text(text: str, model, tokenizer):
    """
    Runs token classification on raw input text using pretrained model with CRF.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
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

    # Filter inputs to match model's forward signature
    signature = inspect.signature(model.forward)
    valid_keys = signature.parameters.keys()
    filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}

    with torch.no_grad():
        logits = model(**filtered_inputs)["logits"]
        mask = filtered_inputs["attention_mask"].bool()

        # Fix CRF mask edge-case
        if mask.size(1) > 0:
            mask[:, 0] = True

        # CRF decode
        predictions = model.crf.decode(logits, mask=mask)

    # Align predictions back to original words
    final_labels = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        label_id = predictions[0][idx]
        final_labels.append((words[word_idx], LABELS[label_id]))
        prev_word_idx = word_idx

    # Postprocess BIO tag errors
    words_only = [w for w, _ in final_labels]
    tags_only = [t for _, t in final_labels]
    corrected_tags = postprocess_bio_tags(tags_only, words_only)

    return list(zip(words_only, corrected_tags))
