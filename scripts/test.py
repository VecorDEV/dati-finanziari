from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = 0 if torch.cuda.is_available() else -1  # Usa GPU se disponibile

model_name = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device if device >= 0 else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def migliora_frase(frase: str) -> str:
    prompt = (
        "Rewrite the following market update as a single fluid, engaging paragraph with a sharp journalistic style. "
        "Do NOT just list facts or separate them with semicolons or short sentences. "
        "Connect ideas smoothly with varied sentence structures and transitions. "
        "Keep all company names, figures, and details exactly as in the original.\n\n"
        "Example: 'Company A rose 10%, while Company B fell 5%, reflecting mixed market sentiment.'\n\n"
        f"Original: {frase}\n\nRewritten:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids.to(device if device >= 0 else "cpu"),
        max_length=200,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise, practical educational trading tip. "
        "Focus on well-known financial indicators or concepts like RSI, Bollinger Bands, VIX, moving averages, or market behaviors. "
        "Explain briefly how or when to use the indicator or behavior in trading decisions. "
        "Do NOT mention specific stocks, numbers or dates.\n\n"
        "Example: 'Tip: Use RSI to spot overbought conditions when it exceeds 70, and oversold when below 30.'\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids.to(device if device >= 0 else "cpu"),
        max_length=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.5,
        do_sample=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Estraggo solo la parte a partire da "Tip:" se presente
    tip_start = decoded.find("Tip:")
    if tip_start != -1:
        return decoded[tip_start:].strip()
    return decoded

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
