!pip install -q transformers datasets
!pip install seqeval
!pip install accelerate -U
!pip install transformers[torch] -U

import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
from sklearn.metrics import precision_recall_fscore_support

dataset = load_dataset("ncbi_disease", split='train')
model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.features['ner_tags'].feature.num_classes)

def biomedical_tokenizer(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=512)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    
# Apply tokenizer and preprocessing
tokenized_datasets = dataset.map(biomedical_tokenizer, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    save_total_limit=2,
    fp16=False,
    load_best_model_at_end=True
)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[dataset.features['ner_tags'].feature.names[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[dataset.features['ner_tags'].feature.names[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()
evaluation_results = trainer.evaluate()


model.save_pretrained("./biomedical_ner_model")
tokenizer.save_pretrained("./biomedical_ner_model")

print("Evaluation results:", evaluation_results)

unique_labels = set()
for example in dataset["ner_tags"]:
    unique_labels.update(example)
print("Unique labels in dataset:", unique_labels)

























