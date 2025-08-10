from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

device = -1  # CPU

model_name = "josty11/distil2"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

paraphraser = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def migliora_frase(frase: str) -> str:
    prompt = (
        "Rewrite the following market update in a fluent, journalistic style, "
        "connecting the ideas naturally and avoiding simple lists or repeated sentence structures. "
        "Keep all the information, numbers, and company names exactly as is, but make the text smooth, engaging, and professional.\n\n"
        f"{frase}\n\nRewritten:"
    )
    results = paraphraser(
        prompt,
        max_new_tokens=160,
        num_return_sequences=1,
        num_beams=6,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return results[0]['generated_text'].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, write ONE concise, precise and practical educational trading tip. "
        "Focus on well-known financial indicators or concepts like RSI, Bollinger Bands, VIX, moving averages, or market behaviors. "
        "Explain briefly how or when to use the indicator or behavior in trading decisions. "
        "Do NOT mention specific stocks, numbers or dates. Make sure the advice is actionable and useful for investors.\n\n"
        "Examples of tips:\n"
        "1. Tip: The Relative Strength Index (RSI) helps identify overbought or oversold conditions, signaling possible trend reversals.\n"
        "2. Tip: Bollinger Bands measure volatility and can indicate when prices may revert after touching the bands.\n"
        "3. Tip: The VIX index reflects market volatility; a rising VIX often signals increased risk and uncertainty.\n"
        "4. Tip: Moving averages smooth out price fluctuations and help spot trend direction and key support or resistance levels.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids,
        max_new_tokens=90,
        num_beams=1,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
