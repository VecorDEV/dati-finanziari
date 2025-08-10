from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

device = -1  # CPU, cambia a 0 per GPU se disponibile

model_name = "tiiuae/falcon-7b-instruct"

print("Caricamento modello e tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # se 8bit serve bitsandbytes
    low_cpu_mem_usage=True,
    device_map="auto" if device != -1 else None,
)

def generate_text(prompt, max_new_tokens=150, num_beams=3, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_new_tokens,
        do_sample=True,
        num_beams=num_beams,
        temperature=temperature,
        no_repeat_ngram_size=3,
        early_stopping=True,
        bad_words_ids=[[tokenizer.unk_token_id]],
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def is_too_generic(text):
    generic_phrases = [
        "stocks go up", "stocks go down", "market continued", "investors reacted",
        "the market is", "the session closes", "mixed day", "the market"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in generic_phrases) or len(text) < 30

def generate_with_retry(prompt_func, max_retries=5):
    for _ in range(max_retries):
        result = prompt_func()
        if not is_too_generic(result):
            return result
    return result  # anche se generic, ritorna l'ultimo tentativo

def migliora_frase(frase: str) -> str:
    prompt = (
        "Rewrite the following market update in a fluent, journalistic style, "
        "connecting ideas naturally and avoiding simple lists or repeated sentences. "
        "Keep all the information, numbers, and company names exactly as is, but make the text smooth, engaging, and professional.\n\n"
        f"{frase}\n\nRewritten:"
    )
    return generate_text(prompt, max_new_tokens=180, num_beams=4, temperature=0.7)

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise, precise and practical educational trading tip. "
        "Focus on well-known financial indicators or concepts like RSI, Bollinger Bands, VIX, moving averages, or market behaviors. "
        "Explain briefly how or when to use the indicator or behavior in trading decisions. "
        "Do NOT mention specific stocks, numbers or dates. Make sure the advice is actionable and useful for investors.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    return generate_text(prompt, max_new_tokens=90, num_beams=2, temperature=0.6)

def migliora_frase_retry(frase):
    return generate_with_retry(lambda: migliora_frase(frase))

def genera_mini_tip_retry(summary):
    return generate_with_retry(lambda: genera_mini_tip_from_summary(summary))

# === Esempio ===
brief_text = (
    "A mixed day in the market with gains balanced by some losses. "
    "Duolingo extended its upward momentum, climbing 23.9%; "
    "ADP extended its downward momentum, jumping 23.4%; "
    "Netflix extended its uptrend, climbing 22.8%; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase_retry(brief_text)
print("Journalistic rewrite:", brief_text_ai)

mini_tip = genera_mini_tip_retry(brief_text_ai)
print("Educational tip:", mini_tip)
