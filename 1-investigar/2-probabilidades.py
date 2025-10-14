"""
Ejemplo 2: Cálculo de Probabilidades

Muestra cómo un LLM calcula probabilidades para predecir el siguiente token.
Modifica la variable PROMPT para experimentar con otras frases.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "La tecnología avanza rápidamente"

print("Cargando modelo y tokenizador...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

print(f"\nTexto: {PROMPT}")

# Tokenizar
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
print(f"Tokens: {input_ids[0].tolist()}")

# Obtener logits
output = model(input_ids)
print(f"\nShape de logits: {output.logits.shape}")
print(f"  [batch_size, secuencia, vocabulario]")

# Logits del último token
final_logits = output.logits[0, -1]

# Argmax: token más probable
argmax = final_logits.argmax()
print(f"\nToken más probable (argmax): {argmax}")
print(f"  → {tokenizer.decode(argmax)}")

# Top-10 candidatos
print(f"\nTop 10 candidatos:")
top10 = torch.topk(final_logits.softmax(dim=0), 10)
for value, index in zip(top10.values, top10.indices):
    print(f"  {tokenizer.decode(index):<10} {value.item():.2%}")
