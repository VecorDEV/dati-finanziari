from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

device = -1  # CPU

model_name = "google/flan-t5-large"

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
        "Do not omit any details or remove any stock names or figures. "
        "The rewritten text must keep all data points and trends exactly, just improve tone and fluency. "
        "Make the sentences concise but ensure all info is preserved.\n\n"
        f"{frase}\n\nRewritten:"
    )
    results = paraphraser(
        prompt,
        max_new_tokens=160,
        num_return_sequences=1,
        num_beams=6,
        do_sample=False,   # <- sampling disabilitato con beam search
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.3  # non serve se do_sample=False, ma puoi lasciarlo
    )
    return results[0]['generated_text'].strip()

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on the market summary below, "
        "write ONE clear, specific educational trading tip about a financial indicator, concept, or market behavior. "
        "Examples include how to use the RSI, Bollinger Bands, or VIX in trading decisions. "
        "Do NOT refer to numbers, percentages, dates, or company names from the summary. Be creative and precise. "
        "Focus on practical, actionable advice suitable for beginner traders, written in plain English. "
        "Start the tip with 'Tip:'.\n\n"
        f"Market summary: {summary}\n\nTip:"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        input_ids,
        max_new_tokens=90,
        num_beams=1,    # <- nessun beam search con sampling attivo
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
    "ADP extended its upward momentum, climbing 23.4%; "
    "Netflix extended its upward momentum, climbing 22.8%; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase(brief_text)
print("Journalistic rewrite:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text_ai)
print("Educational tip:", mini_tip)
