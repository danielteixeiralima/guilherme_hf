# app.py

# Importar as Bibliotecas
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

HUGGINGFACE_TOKEN = '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'

if not HUGGINGFACE_TOKEN:
    raise ValueError("Por favor, forneça o token do Hugging Face em HUGGINGFACE_TOKEN.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Diretório onde o modelo fine-tunado foi salvo (após o treinamento)
# Normalmente, o melhor modelo é salvo no diretório principal definido em output_dir
# Ex: "./checkpoints"
model_dir = "./checkpoints"

# Carregar o tokenizer e o modelo do diretório local, onde o modelo treinado está salvo
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
config = AutoConfig.from_pretrained(model_dir)

# Texto da transcrição (exemplo)
transcription = """
boa noite meu amor, durma bem
"""

if transcription.strip():
    encoding = tokenizer.encode_plus(
        transcription.strip(),
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits.detach().cpu().numpy()[0]
        # Aplicamos softmax aos logits para obter as probabilidades
        probs = softmax(logits)
        # Assumindo: 0: NONTOXIC, 1: TOXIC
        predicted_label = np.argmax(probs)

    if predicted_label == 1:
        abuso_detectado = True
        tipo_abuso = "Texto classificado como ofensivo/toxic"
        explicacao_abuso = "O texto foi classificado como tóxico pelo modelo."
    else:
        abuso_detectado = False
        tipo_abuso = None
        explicacao_abuso = None
else:
    abuso_detectado = False
    tipo_abuso = None
    explicacao_abuso = None

print("Abuso detectado:", abuso_detectado)
