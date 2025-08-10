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
    prompt = (
        "Rewrite the following market update in a fluent, journalistic style, "
        "connecting ideas naturally without listing them. Keep all info intact.\n\n"
        f"{frase}\n\nRewritten:"
    )
    results = generator(
        prompt,
        max_new_tokens=160,
        num_return_sequences=1,
        num_beams=6,
        do_sample=True,
        temperature=0.7,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return results[0]['generated_text'].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise and practical trading tip "
        "about a known indicator or market behavior, without referring to specific stocks or numbers.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    results = generator(
        prompt,
        max_new_tokens=90,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return results[0]['generated_text'].strip()

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
