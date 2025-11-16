Sentiment Analysis with Transformers 🧠💬

A simple, end-to-end Sentiment Analysis project built using PyTorch, HuggingFace Transformers, and the Datasets library.
This project trains a text classification model on custom data and evaluates its performance, with support for inference using HuggingFace pipelines.

🚀 Features

Load and preprocess datasets using 🤗 datasets

Tokenize text using pretrained Transformer tokenizers

Fine-tune a state-of-the-art model (BERT/RoBERTa/DistilBERT, etc.)

Train using HuggingFace Trainer

Evaluate and visualize performance

Run predictions using a simple inference wrapper

🛠️ Tech Stack

Python 3

PyTorch

Transformers (HuggingFace)

Datasets (HuggingFace)

scikit-learn

pandas, numpy

📂 Project Structure
Sentiment_Analysis/
│
├── notebook.ipynb      # Your main training notebook
├── data/               # (Optional) Training dataset
├── models/             # Saved models/checkpoints
├── README.md           # You are here
└── requirements.txt    # Dependencies list

🔧 Setup Instructions
1️⃣ Install Dependencies
pip install torch transformers datasets scikit-learn pandas numpy

2️⃣ Verify GPU (optional but recommended)
import torch
print(torch.cuda.is_available())

🏋️ Training the Model

The training script (inside the notebook) does the following:

✔ Load and split dataset

Using train-test split from scikit-learn.

✔ Tokenize text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

✔ Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

✔ Define training configuration
training_args = TrainingArguments(
    output_dir="model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3
)

✔ Train using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()

📊 Evaluation

After training:

metrics = trainer.evaluate()
print(metrics)


You get:

Accuracy

Precision/recall/F1

Loss curves (if plotted)

🔮 Running Inference
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="model_output")

pipe("This product is amazing!")

📦 Model Export

Fine-tuned model is automatically stored under:

model_output/


You can push it to HuggingFace Hub if needed.

📝 Notes

Works with any binary sentiment dataset.

GPU gives a massive speed boost.

You can switch models by replacing "bert-base-uncased" with anything else (e.g., "distilbert-base-uncased").

❤️ Author

Nayan Kshitij
Cybersecurity + AI enthusiast building cool ML stuff.
