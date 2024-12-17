# train_model.py

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

############################################
# Carregar e processar hatecheck-portuguese
############################################
hc = load_dataset("Paul/hatecheck-portuguese")
print("Splits hatecheck-portuguese:", hc.keys())

if "train" not in hc:
    hc_split = hc["test"].train_test_split(test_size=0.1, seed=42)
else:
    hc_split = hc

def map_hatecheck(example):
    text = example["test_case"]
    label_str = example["label_gold"]  # "hateful" ou "non-hateful"
    label = 1 if label_str == "hateful" else 0
    return {"text": text, "labels": label}

hc_train = hc_split["train"].map(map_hatecheck).remove_columns([c for c in hc_split["train"].column_names if c not in ["text","labels"]])
hc_test = hc_split["test"].map(map_hatecheck).remove_columns([c for c in hc_split["test"].column_names if c not in ["text","labels"]])

hc_dataset = DatasetDict({
    "train": hc_train,
    "test": hc_test
})

############################################
# Carregar e processar hatebr
############################################
hatebr = load_dataset("ruanchaves/hatebr")

def map_hatebr(example):
    text = example["instagram_comments"]
    label_bool = example["offensive_language"]
    label = 1 if label_bool else 0
    return {"text": text, "labels": label}

hb_train = hatebr["train"].map(map_hatebr).remove_columns([c for c in hatebr["train"].column_names if c not in ["text","labels"]])
hb_test = hatebr["test"].map(map_hatebr).remove_columns([c for c in hatebr["test"].column_names if c not in ["text","labels"]])

hb_dataset = DatasetDict({
    "train": hb_train,
    "test": hb_test
})

############################################
# Carregar e processar toxic-text
############################################
tt = load_dataset("nicholasKluge/toxic-text", split="portuguese")

def map_toxic_text(example):
    toxic_str = example["toxic"]
    non_toxic_str = example["non_toxic"]
    if toxic_str is not None and len(toxic_str.strip()) > 0:
        text = toxic_str
        label = 1
    else:
        text = non_toxic_str
        label = 0
    return {"text": text, "labels": label}

tt_processed = tt.map(map_toxic_text).remove_columns(
    [c for c in tt.column_names if c not in ["toxic", "non_toxic"]]+["toxic","non_toxic"]
)

tt_split = tt_processed.train_test_split(test_size=0.1, seed=42)

tt_dataset = DatasetDict({
    "train": tt_split["train"],
    "test": tt_split["test"]
})

############################################
# Unir todos os datasets
############################################
train_combined = concatenate_datasets([hc_dataset["train"], hb_dataset["train"], tt_dataset["train"]])
test_combined = concatenate_datasets([hc_dataset["test"], hb_dataset["test"], tt_dataset["test"]])

final_dataset = DatasetDict({
    "train": train_combined,
    "test": test_combined
})

print("Tamanho final do conjunto de treino:", len(final_dataset["train"]))
print("Tamanho final do conjunto de teste:", len(final_dataset["test"]))

############################################
# Treinar o modelo
############################################
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = final_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Salvar o melhor modelo no diretório ./checkpoints
trainer.save_model("./checkpoints")

print("Treinamento concluído e melhor modelo salvo em ./checkpoints!")
