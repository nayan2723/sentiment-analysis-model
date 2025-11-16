# Sentiment Analysis with Transformers 🧠💬

A simple, end-to-end **Sentiment Analysis** project built using **PyTorch**, **HuggingFace Transformers**, and the **Datasets** library.
This project trains a text classification model on custom data and evaluates its performance, with support for inference using HuggingFace pipelines.

---

## 🚀 Features

* Load and preprocess datasets using 🤗 *datasets*
* Tokenize text using pretrained Transformer tokenizers
* Fine-tune a state-of-the-art model (BERT/RoBERTa/DistilBERT, etc.)
* Train using HuggingFace `Trainer`
* Evaluate and visualize performance
* Run predictions using a simple inference wrapper

---

## 🛠️ Tech Stack

* **Python 3**
* **PyTorch**
* **Transformers (HuggingFace)**
* **Datasets (HuggingFace)**
* **scikit-learn**
* **pandas, numpy**

---

## 📂 Project Structure

```
Sentiment_Analysis/
│
├── notebook.ipynb      # Main training notebook
├── data/               # (Optional) Training dataset
├── models/             # Saved models/checkpoints
├── README.md           # Project documentation
└── requirements.txt    # Dependency list
```

---

## 🔧 Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

### 2️⃣ Verify GPU Availability

```python
import torch
print(torch.cuda.is_available())
```

---

## 🏋️ Training the Model

### ✔ Load and split dataset

### ✔ Tokenize text

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### ✔ Load model

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### ✔ Define training configuration

```python
training_args = TrainingArguments(
    output_dir="model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3
)
```

### ✔ Train using Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()
```

---

## 📊 Evaluation

```python
metrics = trainer.evaluate()
print(metrics)
```

Outputs include:

* Accuracy
* Precision / Recall / F1
* Loss curves

---

## 🔮 Running Inference

```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="model_output")

pipe("This product is amazing!")
```

---

## 📦 Model Export

Fine-tuned model is saved automatically under:

```
model_output/
```

---

## ❤️ Author

**Nayan Kshitij**
Cybersecurity & AI Enthusiast
