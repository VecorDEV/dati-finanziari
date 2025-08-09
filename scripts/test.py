from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration



#MODELLO 1 DI IA PER RAFFINAMENTO FRASI
model_name_paraphrase = "ramsrigouthamg/t5_paraphraser"

tokenizer_paraphrase = AutoTokenizer.from_pretrained(model_name_paraphrase, use_fast=False)
model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained(model_name_paraphrase)

paraphraser = pipeline("text2text-generation", model=model_paraphrase, tokenizer=tokenizer_paraphrase)

def migliora_frase(frase: str) -> str:
    """
    Accetta una frase in input e restituisce una versione migliorata sintatticamente
    e stilisticamente tramite parafrasi.
    """
    risultati = paraphraser(frase, max_length=100, num_return_sequences=1)
    frase_migliorata = risultati[0]['generated_text']
    return frase_migliorata


# MODELLO 2 DI IA PER GENERAZIONE TIPS
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def genera_mini_tip_from_summary(summary: str) -> str:
    # Prompt per far capire al modello cosa vogliamo
    input_text = (
        f"From the following financial market summary, write one or two short tips or explanations "
        f"about important terms or concepts mentioned, to help a non-expert reader understand better:\n"
        f"{summary}"
    )
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=80,
        num_beams=2,
        early_stopping=True
    )
    tip = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tip


brief_text_ai = migliora_frase(brief_text)
print("Frase migliorata:", brief_text_ai)

mini_tip = genera_mini_tip_from_summary(brief_text)
print("Mini tip generato:", mini_tip)
