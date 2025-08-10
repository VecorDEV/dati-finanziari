from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

device = 0 if torch.cuda.is_available() else -1  # Usa GPU se disponibile

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)

paraphraser = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def migliora_frase(frase: str) -> str:
    prompt = (
        "Rewrite the following market update in a fluent, journalistic style, "
        "connecting ideas naturally without simple lists or repetition. "
        "Keep all numbers and company names unchanged.\n\n"
        f"{frase}\n\nRewrite:"
    )
    results = paraphraser(
        prompt,
        max_new_tokens=160,
        num_return_sequences=1,
        num_beams=6,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return results[0]['generated_text'].split("Rewrite:")[-1].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise and practical educational trading tip. "
        "Focus on well-known indicators like RSI, Bollinger Bands, VIX, moving averages or market behaviors. "
        "Do NOT mention specific stocks, numbers, or dates.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    results = paraphraser(
        prompt,
        max_new_tokens=90,
        num_return_sequences=1,
        num_beams=4,
        do_sample=True,
        temperature=0.6,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return results[0]['generated_text'].split("Tip:")[-1].strip()

# === Esempio ===
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
