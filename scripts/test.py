from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Parafrasatore per migliorare il summary mantenendo il contenuto ===
model_name_paraphrase = "Vamsi/T5_Paraphrase_Paws"
tokenizer_paraphrase = AutoTokenizer.from_pretrained(model_name_paraphrase)
model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained(model_name_paraphrase)
paraphraser = pipeline("text2text-generation", model=model_paraphrase, tokenizer=tokenizer_paraphrase)

def migliora_frase(frase: str) -> str:
    risultati = paraphraser(
        frase,
        max_new_tokens=200,        # aumenta un po' per evitare tagli
        num_return_sequences=1,    # tieni solo 1 per coerenza
        num_beams=8,                # beam search più alta per fedeltà
        do_sample=False,            # disattiva sampling per evitare invenzioni
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return risultati[0]['generated_text']

# === Modello per generare il mini tip didattico ===
model_name_tip = "google/flan-t5-large"
tokenizer_tip = AutoTokenizer.from_pretrained(model_name_tip)
model_tip = AutoModelForSeq2SeqLM.from_pretrained(model_name_tip)

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "You are a financial educator. Based on this market summary or your financial knowledge, "
        "write one short, specific, and educational tip that explains a financial concept, indicator, "
        "or market behavior. The tip should be relevant and useful for beginner investors. "
        "Examples: explain what RSI means, how to read Bollinger Bands, why the VIX is called the fear index, "
        "or when to buy based on VIX trends.\n\n"
        f"Market summary: {summary}\nTip:"
    )
    input_ids = tokenizer_tip.encode(prompt, return_tensors="pt", truncation=True)
    outputs = model_tip.generate(
        input_ids,
        max_new_tokens=120,
        num_beams=6,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    tip = tokenizer_tip.decode(outputs[0], skip_special_tokens=True)
    return tip

# === Esempio di input ===
brief_text = (
    "A mixed day in the market with gains balanced by some losses. "
    "Duolingo extended its upward momentum, climbing 23.9%; "
    "ADP extended its upward momentum, climbing 23.4%; "
    "Netflix extended its upward momentum, climbing 22.8%; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase(brief_text)
print("Frase migliorata:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text_ai)
print("Mini tip generato:", mini_tip)
