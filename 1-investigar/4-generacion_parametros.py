"""
Ejemplo 4: Generación con parámetros

Incluye los hiperparámetros configurables directamente en el código para
experimentar con diferentes estilos de salida.

Uso:
  python 4-generacion_parametros.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompt inicial (modifica libremente)
PROMPT = "Una tarde lluviosa invitaba a quedarse en casa"

# Parámetros de generación (ajusta y vuelve a ejecutar)
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8          # < 1.0 = más determinista, > 1.0 = más creativo
TOP_K = 40                 # 0 desactiva top-k
TOP_P = 0.9                # 1.0 desactiva nucleus sampling
REPETITION_PENALTY = 1.1   # 1.0 no aplica penalización

print("Cargando modelo y tokenizador...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

print(f"\nPrompt: {PROMPT!r}")
inputs = tokenizer(PROMPT, return_tensors="pt")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    do_sample=True,
    repetition_penalty=REPETITION_PENALTY,
)

texto = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\nSalida completa:")
print(texto)

