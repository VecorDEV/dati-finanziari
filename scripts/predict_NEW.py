import spacy

# Carica il modello linguistico inglese
nlp = spacy.load("en_core_web_sm")

# Funzione per classificare le parole
def classify_words(text):
    # Analizza il testo con spaCy
    doc = nlp(text)
    
    # Liste per ciascuna categoria grammaticale
    nouns = []
    verbs = []
    adjectives = []
    adverbs = []
    
    # Itera attraverso ogni parola nel testo
    for token in doc:
        # Classifica il token in base al suo Part of Speech (POS)
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "ADJ":
            adjectives.append(token.text)
        elif token.pos_ == "ADV":
            adverbs.append(token.text)
    
    # Ritorna i risultati
    return nouns, verbs, adjectives, adverbs

# Test del codice con la frase fornita
text = """Analysts anticipate that Tesla’s first-quarter vehicle deliveries will be the lowest in over two years, with estimates around 377,592 units. This projection is attributed to several factors, including production challenges with the Model Y, public backlash against CEO Elon Musk’s political involvement, and consumer delays in anticipation of a lower-priced model. Stifel analyst Stephen Gengaro reduced his delivery forecast by 23%, highlighting concerns over Musk’s association with President Donald Trump and its impact on sales."""

nouns, verbs, adjectives, adverbs = classify_words(text)

# Stampa i risultati
print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)
print("Adverbs:", adverbs)
