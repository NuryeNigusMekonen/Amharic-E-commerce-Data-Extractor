##  Amharic Named Entity Recognition (NER)

A deep learning pipeline for Named Entity Recognition on **Amharic e-commerce data**, using multilingual Transformer models with **CRF decoding**, **weighted loss for class imbalance**, and **interpretability tools** like **SHAP** and **LIME**.

---

##  Project Structure

```
Amharic-E-commerce-Data-Extractor/
│
├── data/                       # Raw and labeled data (e.g., .conll format)
├── models/                    # Saved model checkpoints
├── notebooks/                 # Interactive notebooks for training and experiments
├── src/                       # Main source code
│   ├── config.py              # Config: paths, labels, model list, weights
│   ├── model.py               # CRF-enhanced Transformer model
│   ├── prepare_dataset.py     # Dataset loading and preprocessing
│   ├── train.py               # Model training script
│   ├── predict.py             # Inference and post-processing
│   ├── evaluate_models.py     # Model comparison and selection
│   └── interpret.py           # SHAP & LIME interpretability
```

---

##  Features

*  **Multilingual Transformers**: `XLM-R`, `mBERT`, `AfroXLMR`, `BERT-Tiny-Amharic`
*  **CRF Layer** for better sequence modeling
*  **Weighted Loss** to handle label imbalance
*  **BIO Tag Postprocessing** to fix tagging errors
*  **Model Comparison**: Accuracy, F1, and robustness
*  **Explainability**: LIME & SHAP for token-level insights

---

##  Installation

```bash
git clone https://github.com/your-username/Amharic-E-commerce-Data-Extractor.git
cd Amharic-E-commerce-Data-Extractor

# Create and activate virtual environment
python -m venv AE-venv-py310
source AE-venv-py310/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Labels

The NER system supports 9 classes (BIO format):

* `B-PRODUCT`, `I-PRODUCT`
* `B-PRICE`, `I-PRICE`
* `B-LOC`, `I-LOC`
* `B-PHONE`, `I-PHONE`
* `O` – Outside entity

---

##  Training

Train and save models with CRF layer:

```bash
python src/train.py
```

You can edit the `MODEL_LIST` in `config.py` to train multiple models sequentially.

---

##  Evaluation & Model Selection

Run the model comparison tool:

```bash
python src/evaluate_models.py
```

This compares F1 scores and prints detailed `seqeval` classification reports.

---

##  Prediction

To predict tags for a sentence using a saved model:

```python
from src.predict import load_model, predict_text

model, tokenizer = load_model("xlm-roberta-base", "models/xlm-roberta-ner/checkpoint-200")
text = "ዋጋ 200 ብር ነው የእቃው"
result = predict_text(text, model, tokenizer)
print(result)
```

---

##  Interpretability

### 1. SHAP Explanation

```python
from src.interpret import shap_explanation, display_shap

shap_values, words = shap_explanation(text, model, tokenizer)
display_shap(shap_values, words)
```

### 2. LIME Explanation

```python
from lime.lime_text import LimeTextExplainer
from src.interpret import lime_forward_fn

explainer = LimeTextExplainer(class_names=LABELS)
exp = explainer.explain_instance(text, lime_forward_fn, labels=[1])
exp.show_in_notebook()
```

---

##  Example Input & Output

### Input:

```text
ዋጋ 200 ብር ነው የእቃው
```

### Output:

```
[('ዋጋ', 'B-PRICE'), ('200', 'I-PRICE'), ('ብር', 'I-PRICE'), ('ነው', 'O'), ('የእቃው', 'B-PRODUCT')]
```

---

##  Model Performance (F1-score)

| Model                  | F1 Score |
| ---------------------- | -------- |
| xlm-roberta-base       | 0.42     |
| bert-tiny-amharic      | 0.23     |
| afro-xlmr-base         | 0.00     |
| bert-base-multilingual | 0.11     |

 **Best Model**: `xlm-roberta-base`

---

##  Metrics

Evaluation is based on:

* Entity-level **Precision**, **Recall**, and **F1-score**
* Handled via `seqeval` and `classification_report`

---

##  Future Improvements

* Augment training data for rare entities
* Add user feedback loop
* Deploy via Streamlit for interactive annotation

---

##  License

MIT License. See [LICENSE](LICENSE).

---

##  Acknowledgements

* [HuggingFace Transformers](https://huggingface.co)
* [TorchCRF](https://github.com/kmkurn/pytorch-crf)
* [10 Academy](https://10academy.org) for challenge support

---