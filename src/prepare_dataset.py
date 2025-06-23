import os
from collections import Counter
from datasets import Dataset, ClassLabel, Features, Sequence, Value
from transformers import AutoTokenizer
from src.config import CONLL_PATH, LABELS


def read_conll_file(path):
    """Reads a CoNLL file and returns a list of token/NER tag sequences."""
    sequences = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens and tags:
                    sequences.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    token, tag = splits[0], splits[-1]
                    tokens.append(token)
                    tags.append(tag)
    if tokens and tags:
        sequences.append({"tokens": tokens, "ner_tags": tags})
    return sequences
def get_dataset():
    """Parses the CoNLL data and returns a HuggingFace Dataset with proper features."""
    sequences = read_conll_file(CONLL_PATH)
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    all_tags = []
    for seq in sequences:
        seq["ner_tags"] = [label2id[tag] for tag in seq["ner_tags"]]
        all_tags.extend(seq["ner_tags"])
    print("\nLabel distribution:")
    for tag_id, count in Counter(all_tags).items():
        print(f"{id2label[tag_id]}: {count}")
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    dataset = Dataset.from_list(sequences)
    dataset = dataset.cast(features)
    return dataset.train_test_split(test_size=0.2)
def tokenize_and_align(dataset, model_name):
    """Tokenizes input and aligns word-level NER tags with tokenized subwords."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def align_batch(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        all_labels = []
        for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
            word_labels = batch["ner_tags"][i]
            label_ids = []

            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(word_labels[word_idx])
                prev_word_idx = word_idx

            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        tokenized.pop("offset_mapping")
        return tokenized

    tokenized_dataset = dataset.map(align_batch, batched=True)
    return tokenized_dataset, tokenizer
