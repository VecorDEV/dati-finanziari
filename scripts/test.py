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
        "Rewrite this market update in a fluent, journalistic style:\n\n"
        f"{frase}\n\nRewritten:"
    )
    results = generator(
        prompt,
        max_new_tokens=150,
        num_return_sequences=1,
        num_beams=5,
        do_sample=False,           # beam search senza sampling
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    text = results[0]['generated_text']
    # Rimuovi il prompt iniziale dal testo generato
    return text.split("Rewritten:")[-1].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise and practical trading tip "
        "about a known indicator or market behavior, without mentioning specific stocks or numbers:\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    results = generator(
        prompt,
        max_new_tokens=90,
        num_return_sequences=1,
        do_sample=True,            # sampling per maggiore creatività
        temperature=0.7,
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
