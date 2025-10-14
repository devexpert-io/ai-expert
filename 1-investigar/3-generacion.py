"""
Ejemplo 3: Generación de Texto

Produce una continuación corta a partir de un prompt inicial.
Modifica la variable PROMPT para generar nuevos textos.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "El amanecer sobre la ciudad"

print("Cargando modelo y tokenizador...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

inputs = tokenizer(PROMPT, return_tensors="pt")

print(f"\nPrompt: {PROMPT!r}")
generated_ids = model.generate(**inputs, max_new_tokens=40)
texto = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\nSalida completa:")
print(texto)
