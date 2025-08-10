from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = -1  # CPU

model_name = "google/flan-t5-large"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def migliora_frase(frase: str) -> str:
    prompt = (
        "Rewrite the following market update in a fluent, journalistic style, "
        "connecting the ideas naturally and avoiding simple lists or repeated sentence structures. "
        "Keep all the information, numbers, and company names exactly as is, but make the text smooth, engaging, and professional.\n\n"
        f"{frase}\n\nRewritten:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 180,  # input + output max tokens
        do_sample=False,                      # no sampling per stabilità
        num_beams=4,                         # beam search per qualità
        no_repeat_ngram_size=3,
        early_stopping=True,
        bad_words_ids=[[tokenizer.unk_token_id]]  # evita token <unk>
    )
    # Prendo solo la parte generata (escludo il prompt)
    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise, precise and practical educational trading tip. "
        "Focus on well-known financial indicators or concepts like RSI, Bollinger Bands, VIX, moving averages, or market behaviors. "
        "Explain briefly how or when to use the indicator or behavior in trading decisions. "
        "Do NOT mention specific stocks, numbers or dates. Make sure the advice is actionable and useful for investors.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 90,
        do_sample=False,
        num_beams=2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        bad_words_ids=[[tokenizer.unk_token_id]]
    )
    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
