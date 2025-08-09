from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration


# MODELLO 1 DI IA PER RAFFINAMENTO FRASI
model_name_paraphrase = "ramsrigouthamg/t5_paraphraser"

tokenizer_paraphrase = AutoTokenizer.from_pretrained(model_name_paraphrase, use_fast=False)
model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained(model_name_paraphrase)

paraphraser = pipeline(
    "text2text-generation",
    model=model_paraphrase,
    tokenizer=tokenizer_paraphrase
)

def migliora_frase(frase: str) -> str:
    """
    Accetta una frase in input e restituisce una versione migliorata sintatticamente
    e stilisticamente tramite parafrasi, con parametri di generazione migliorati.
    """
    risultati = paraphraser(
        frase,
        max_length=150,        # più lunghezza per output più ricco
        num_return_sequences=3, # più alternative generate
        num_beams=5,            # beam search più ampia per qualità
        do_sample=True,         # sampling per varietà
        temperature=0.8         # temperatura moderata per creatività
    )
    # Prendi la prima parafrasi generata (puoi cambiarla se vuoi)
    frase_migliorata = risultati[0]['generated_text']
    return frase_migliorata


# MODELLO 2 DI IA PER GENERAZIONE TIPS
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def genera_mini_tip_from_summary(summary: str) -> str:
    # Prompt migliorato per chiarezza
    input_text = (
        f"Write one or two concise tips explaining important financial terms or events from this summary, "
        f"for a beginner reader:\n{summary}"
    )
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=100,      # output più lungo
        num_beams=5,         # beam search più esteso per qualità
        do_sample=True,      # sampling per creatività
        temperature=0.7,     # temperatura moderata
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

mini_tip = genera_mini_tip_from_summary(brief_text)
print("Mini tip generato:", mini_tip)
