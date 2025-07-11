LABELS = [
    "O", "B-PRODUCT", "I-PRODUCT",
    "B-PRICE", "I-PRICE",
    "B-LOC", "I-LOC",
    "B-PHONE", "I-PHONE"
]

LABEL_COUNTS = [10000, 80, 60, 100, 75, 50, 45, 40, 30]
CLASS_WEIGHTS = [1.0 / c for c in LABEL_COUNTS]
CLASS_WEIGHTS = [w / sum(CLASS_WEIGHTS) for w in CLASS_WEIGHTS]

MODEL_LIST = {
    "xlm-roberta-base": "../models/xlm-roberta-ner/checkpoint-200",
    "rasyosef/bert-tiny-amharic": "../models/bert-tiny-amharic/checkpoint-200",
    "Davlan/afro-xlmr-base": "../models/afroxlmr-ner/checkpoint-200",
    "google-bert/bert-base-multilingual-cased": "../models/mbert-ner/checkpoint-200"
}
CONLL_PATH = "../data/labeled/ner_auto_labels.conll"
