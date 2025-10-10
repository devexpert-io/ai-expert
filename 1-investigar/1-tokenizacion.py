"""
Ejemplo 1: Tokenización

Muestra cómo un modelo de lenguaje divide el texto en tokens.

Uso: python tokenizacion.py "tu texto aquí"
"""

import sys
from transformers import AutoTokenizer

prompt = "".join(sys.argv[1:])

# Cambia el tokenizador para ver cómo modifica el resultado

tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")

tokens = tokenizer(prompt).input_ids

for token_id in tokens:
    print(f"  {token_id} → {tokenizer.decode([token_id])}")