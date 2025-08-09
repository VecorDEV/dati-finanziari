from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration


# MODELLO 1 DI IA PER RAFFINAMENTO FRASI (parafrasi)
model_name_paraphrase = "Vamsi/T5_Paraphrase_Paws"

tokenizer_paraphrase = AutoTokenizer.from_pretrained(model_name_paraphrase)
model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained(model_name_paraphrase)

paraphraser = pipeline(
    "text2text-generation",
    model=model_paraphrase,
    tokenizer=tokenizer_paraphrase
)

def migliora_frase(frase: str) -> str:
    risultati = paraphraser(
        frase,
        max_new_tokens=150,        # usa max_new_tokens per evitare warning
        num_return_sequences=3,
        num_beams=5,
        do_sample=True,
        temperature=0.8,
        no_repeat_ngram_size=2
    )
    # Prendi la prima parafrasi
    return risultati[0]['generated_text']


# MODELLO 2 DI IA PER GENERAZIONE MINI TIP (t5-base)
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def genera_mini_tip_from_summary(summary: str) -> str:
    prompt = (
        "Given the following financial market summaries, write one concise and educational tip "
        "explaining an important financial term or event for beginner traders.\n\n"
        "Summary: The stock market showed volatility today with tech stocks leading gains. Apple shares rose by 3%.\n"
        "Tip: Apple’s stock rise indicates strong investor confidence in their latest products.\n\n"
        "Summary: The energy sector dropped due to falling oil prices.\n"
        "Tip: Lower oil prices can reduce profits for energy companies, causing their stocks to fall.\n\n"
        f"Summary: {summary}\nTip:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        num_beams=5,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    tip = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tip


brief_text = (
    "A mixed day in the market with gains balanced by some losses. "
    "Duolingo extended its upward momentum, climbing 23.9%.; "
    "ADP extended its upward momentum, climbing 23.4%.; "
    "Netflix extended its upward momentum, climbing 22.8%.; "
    "McDonald's rose by 23.2%, leading the gains today. "
    "The session closes with a balanced market tone and cautious positioning."
)

brief_text_ai = migliora_frase(brief_text)
print("Frase migliorata:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text_ai)
print("Mini tip generato:", mini_tip)
