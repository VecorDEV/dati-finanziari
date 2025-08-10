from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

device = -1  # CPU

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def migliora_frase(frase: str) -> str:
    prompt = f"Rewrite the following market update in fluent journalistic style:\n\n{frase}\n\nRewrite:"
    results = generator(
        prompt,
        max_new_tokens=120,
        num_return_sequences=1,
        num_beams=5,
        do_sample=False,
        early_stopping=True,
        temperature=0.4,
        no_repeat_ngram_size=3
    )
    text = results[0]['generated_text']
    return text.split("Rewrite:")[-1].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = f"Based on this market summary, write a concise educational trading tip (no stocks or numbers):\n\n{summary}\n\nTip:"
    results = generator(
        prompt,
        max_new_tokens=60,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        no_repeat_ngram_size=3
    )
    text = results[0]['generated_text']
    return text.split("Tip:")[-1].strip()

# Esempio
brief_text = (
    "A mixed day in the market with gains balanced by some losses. "
    "Duolingo extended its upward momentum, climbing 23.9%; "
    "ADP extended its downward momentum, jumping 23.4%; "
    "Netflix extended its uptrend, climbing 22.8%; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase(brief_text)
print("Journalistic rewrite:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text_ai)
print("Educational tip:", mini_tip)
