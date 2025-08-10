from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

device = -1  # CPU

model_name = "google/flan-t5-large"  # modello più potente, migliore qualità output

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
        "Rewrite the following market update with a sharp, professional, and journalistic tone. "
        "Keep it concise, but do not remove information, highlight key price moves and overall market mood. "
        "Avoid adding new facts.\n\n"
        f"{frase}\n\nRewritten:"
    )
    results = paraphraser(
        prompt,
        max_new_tokens=80,
        max_length=200,  # evita taglio a 20 token
        num_return_sequences=1,
        num_beams=6,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return results[0]['generated_text'].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the general market sentiment below, "
        "write ONE short, self-contained educational trading tip. If nothin in the text below is important as advide, give your opinion on a general financial advice you think is important."
        "The tip must explain a financial concept, indicator, or market behavior (e.g., RSI, Bollinger Bands, the VIX, moving averages). "
        "Do NOT include numbers, percentages, or company names from the text. "
        "Focus on timeless principles, not events. "
        "Write in plain English, 1 sentence, starting with 'Tip:'.\n\n"
        f"Market sentiment: {summary}\n\nTip:"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        max_length=200,  # evita taglio anticipato
        num_beams=4,
        do_sample=True,
        temperature=0.8,  # più creativo
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# === Esempio ===
brief_text = (
    "A mixed day in the market with gains balanced by some losses. "
    "Duolingo extended its upward momentum, climbing 23.9%; "
    "ADP extended its upward momentum, climbing 23.4%; "
    "Netflix extended its upward momentum, climbing 22.8%; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase(brief_text)
print("Journalistic rewrite:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text_ai)
print("Educational tip:", mini_tip)
