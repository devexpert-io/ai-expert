"""
Ejemplo 1: Tokenización

Muestra cómo un modelo de lenguaje divide el texto en tokens.
Modifica la variable PROMPT para probar diferentes frases.
"""

from transformers import AutoTokenizer

PROMPT = "Era un día tormentoso y nublado"

# Cambia el tokenizador para ver cómo modifica el resultado.
tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokens = tokenizer(PROMPT).input_ids

for token_id in tokens:
    print(f"  {token_id} → {tokenizer.decode([token_id])}")
