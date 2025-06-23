import pandas as pd
import torch
from transformers import AutoTokenizer
from src.model import TokenClassificationModel
from src.config import LABELS, MODEL_LIST
from safetensors.torch import load_file
from collections import defaultdict
import re
import numpy as np
from datetime import datetime

#  Load Best Performing Model (update path as needed)
def load_best_model(model_name, checkpoint_dir):
    model = TokenClassificationModel(model_name=model_name, num_labels=len(LABELS))
    model.load_state_dict(load_file(f"{checkpoint_dir}/model.safetensors"), strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer

# Extract Entities from Text
def extract_entities(text, model, tokenizer, device):
    words = text.split()
    tokenizer_output = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    word_ids = tokenizer_output.word_ids(batch_index=0)  # Fix: get word_ids from output
    tokenizer_output.pop("offset_mapping", None)  # Optional if not used
    inputs = {k: v.to(device) for k, v in tokenizer_output.items()}
    with torch.no_grad():
        logits = model(**inputs)["logits"]
        mask = inputs["attention_mask"].bool()
        if mask.size(1) > 0:
            mask[:, 0] = True
        preds = model.crf.decode(logits, mask=mask)
    labels = preds[0]
    results = defaultdict(list)
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        label = LABELS[labels[idx]]
        word = words[word_idx]
        if label != "O":
            tag_type = label.split("-")[-1]
            results[tag_type].append(word)
        prev_word_idx = word_idx

    return results


# Score Vendor Channel
def score_vendor(df, model, tokenizer, device):
    post_dates = pd.to_datetime(df["timestamp"], errors="coerce")
    weeks = (post_dates.max() - post_dates.min()).days / 7
    posting_frequency = len(df) / weeks if weeks > 0 else len(df)
    
    avg_views = df["views"].mean()
    top_post = df.loc[df["views"].idxmax()]

    prices = []
    for text in df["message"]:
        entities = extract_entities(text, model, tokenizer, device)
        for p in entities.get("PRICE", []):
            value = re.findall(r"\d+", p)
            if value:
                prices.append(int(value[0]))

    avg_price = np.mean(prices) if prices else 0
    lending_score = (avg_views * 0.5) + (posting_frequency * 0.5)

    return {
        "Posts/Week": round(posting_frequency, 2),
        "Avg. Views/Post": round(avg_views, 2),
        "Avg. Price (ETB)": round(avg_price, 2),
        "Lending Score": round(lending_score, 2),
        "Top Product": " ".join(extract_entities(top_post["message"], model, tokenizer, device).get("PRODUCT", [])),
        "Top Price": " ".join(extract_entities(top_post["message"], model, tokenizer, device).get("PRICE", []))
    }

# Run Scorecard on Dataset
def generate_scorecard(csv_path, model_name, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_best_model(model_name, checkpoint_dir)
    model.to(device)

    df = pd.read_csv(csv_path)
    vendors = df["channel_name"].unique()

    records = []
    for vendor in vendors:
        vendor_df = df[df["channel_name"] == vendor]
        print(f"Scoring {vendor}...")
        record = score_vendor(vendor_df, model, tokenizer, device)
        record["Vendor"] = vendor
        records.append(record)

    scorecard = pd.DataFrame(records)
    scorecard = scorecard[["Vendor", "Posts/Week", "Avg. Views/Post", "Avg. Price (ETB)", "Lending Score", "Top Product", "Top Price"]]
    scorecard.sort_values("Lending Score", ascending=False, inplace=True)
    return scorecard
