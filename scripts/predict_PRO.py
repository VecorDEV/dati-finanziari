import spacy
from github import Github, GithubException
import re
import feedparser
import os

# Carichiamo il modello di lingua inglese
nlp = spacy.load("en_core_web_sm")


def get_stock_news(symbol):
    """ Recupera i titoli delle notizie per un determinato simbolo. """
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    titles = [entry.title for entry in feed.entries]
    return titles


# Dizionario personalizzato di parole con i punteggi di sentiment
sentiment_dict = {
    "achieve": 0.8,  
    "achievement": 0.9,  
    "advantage": 0.7,  
    "advancing": 0.8,  
    "all-time": 1.0,  
    "amazing": 1.0,  
    "appealing": 0.7,  
    "ascending": 0.7,  
    "asset": 0.8,  
    "attractive": 0.8,  
    "audacious": 0.6,  
    "autonomous": 0.7,  
    "alarm": -0.7,  
    "anxiety": -0.8,  
    "atrocious": -1.0,  
    "ailing": -0.7,  
    "adverse": -0.8,  
    "aggravate": -0.8,  
    "abandon": -0.9,  
    "absent": -0.7,  
    "atrophy": -0.9,  
    "alarming": -0.9,  
    "awkward": -0.6,  
    "anti": -0.8,
    "absolutely": 1.5,  # Intensificatore forte
    "adequately": 0.6,  # Moderato, attenuante
    "aggressively": 1.2,  # Intensificatore
    "almost": 0.5,  # Attenuatore
    "angrily": -1.0,  # Intensificatore negativo
    "anxiously": -1.0,  # Intensificatore negativo
    "apparently": 1.0,  # Neutrale, ma può dare una sfumatura positiva o negativa
    "arguably": 1.0,  # Neutrale, ma enfatizza un punto di vista
    "assiduously": 1.2,  # Intensificatore positivo
    "astonishingly": 1.5,  # Intensificatore positivo
    "awkwardly": -0.5,  # Attenuante o negativo
    "adequately": 0.6,  # Attenuatore

    "bull": 0.9,  
    "bullish": 1.0,  
    "bullishness": 1.0,  
    "bulls": 0.9,  
    "bullmarket": 1.0,  
    "bullishtrend": 1.0, 
    "boom": 0.8,  
    "booming": 1.0,  
    "boost": 0.9,  
    "better": 0.8,  
    "brilliant": 1.0,  
    "breakthrough": 1.0,  
    "benefit": 0.7,  
    "balance": 0.6,  
    "bountiful": 1.0,  
    "bright": 0.8,  
    "billion": 0.8,  
    "bonanza": 1.0,  
    "bustling": 0.7,  
    "broadening": 0.6, 
    "bear": -0.9,  
    "bearish": -1.0,  
    "bearishness": -1.0,  
    "bears": -0.9,  
    "bearmarket": -1.0,  
    "bearishtrend": -1.0,  
    "blow": -0.8,  
    "bust": -1.0,  
    "balk": -0.7,  
    "bad": -0.9,  
    "burden": -0.8,  
    "bruising": -0.9,  
    "barren": -0.8,  
    "bitter": -0.7,  
    "bumpy": -0.6,  
    "bankrupt": -1.0,  
    "blunder": -0.9,  
    "block": -0.7,  
    "bleak": -0.8,
    "broadly": 0.6,  # Avverbio che indica una valutazione generale, attenuante
    "briskly": 1.1,  # Intensificatore positivo, indica un'azione rapida e positiva
    "boldly": 1.2,  # Intensificatore positivo, denota coraggio o audacia
    "badly": -1.0,  # Intensificatore negativo
    "bitterly": -1.0,  # Intensificatore negativo
    "broadly": 0.7,  # Intensifica un concetto neutro, ma spesso positivo
    "barely": -0.5,  # Attenuante che riduce l'intensità positiva o negativa
    "brutally": -1.5,  # Intensificatore negativo, esprime un'azione severa
    "boldly": 1.1,  # Intensificatore positivo
    "blatantly": -1.2,  # Intensificatore negativo
    "bullishly": 1.2,  # Intensificatore positivo
    "bearishly": -1.2,  # Intensificatore negativo

    "climb": 0.8,  
    "climbing": 0.9,  
    "confidence": 1.0,  
    "confidenceboost": 1.0,  
    "creative": 0.9,  
    "capital": 0.8,  
    "cashflow": 0.9,  
    "comfortable": 0.7,  
    "clutch": 1.0,  
    "cash": 0.9,  
    "cheerful": 0.8,  
    "cuttingedge": 1.0,  
    "catalyst": 0.8,  
    "committed": 0.8,  
    "consistently": 0.9,  
    "contribute": 0.7,  
    "courageous": 1.0,  
    "clearly": 0.7,  
    "collapse": -1.0,  
    "crash": -1.0,  
    "contraction": -0.8,  
    "costly": -0.9,  
    "caution": -0.7,  
    "crisis": -1.0,  
    "clumsy": -0.8,  
    "chaos": -1.0,  
    "criticism": -0.9,  
    "corruption": -1.0,  
    "cramped": -0.7,  
    "cutback": -0.8,  
    "cliff": -0.9,  
    "complicated": -0.8,  
    "constrained": -0.7,  
    "counterproductive": -0.9,  
    "collapse": -1.0,  
    "convoluted": -0.8,  
    "corrupt": -1.0,  
    "carefully": 0.6,  # Avverbio che indica attenzione e ponderazione, attenuante
    "confidently": 1.1,  # Intensificatore positivo che denota sicurezza
    "cautiously": -0.6,  # Avverbio che attenua una situazione positiva
    "crashingly": -1.2,  # Intensificatore negativo, esprime un crollo grave
    "clearly": 0.7,  # Avverbio positivo che indica chiarezza
    "clumsily": -1.0,  # Intensificatore negativo, denota goffaggine o errori
    "critically": -1.0,  # Intensificatore negativo, indica un problema grave
    "conservatively": -0.5,  # Avverbio che attenua, denota approccio prudente
    "competitively": 0.8,  # Intensificatore positivo, denota competitività
    "courageously": 1.2,  # Intensificatore positivo, denota azione audace

    "dynamic": 0.9,  
    "driving": 0.8,  
    "diligent": 0.9,  
    "diversification": 0.8,  
    "dawn": 0.9,  
    "distinguished": 1.0,  
    "dazzling": 1.0,  
    "dedicated": 0.8,  
    "dominant": 0.9,  
    "dramatic": 0.8,  
    "dynamicgrowth": 1.0,  
    "development": 0.8,  
    "driven": 0.9,  
    "delightful": 0.9,  
    "double": 0.7,  
    "decent": 0.7,  
    "durable": 0.8,  
    "deal": 0.8,  
    "deliver": 0.9,  
    "dividend": 0.8,  
    "decline": -0.9,  
    "downturn": -1.0,  
    "debt": -0.9,  
    "damaging": -1.0,  
    "disaster": -1.0,  
    "downfall": -1.0,  
    "doubt": -0.9,  
    "deficit": -0.9,  
    "delayed": -0.8,  
    "deterioration": -1.0,  
    "disruptive": -0.9,  
    "dismal": -0.9,  
    "difficult": -0.8,  
    "drain": -0.8,  
    "declining": -0.9,  
    "destructive": -1.0,  
    "dismayed": -0.9,  
    "doubtful": -0.8,  
    "dramatically": 1.1,  # Intensificatore positivo, indica un cambiamento significativo
    "diligently": 0.9,  # Intensificatore positivo, denota lavoro costante e applicato
    "desperately": -1.2,  # Intensificatore negativo, indica una condizione critica
    "deliberately": -0.6,  # Avverbio che attenua una situazione, denota ponderazione
    "downwardly": -0.8,  # Intensificatore negativo, denota un movimento verso il basso
    "doubtfully": -0.7,  # Intensificatore negativo, denota incertezza o scetticismo
    "decisively": 0.9,  # Intensificatore positivo, denota decisione e sicurezza
    "dramatically": 1.2,  # Intensificatore positivo, denota un cambiamento improvviso
    "delightfully": 1.0,  # Intensificatore positivo, denota qualcosa di piacevole o gratificante

    "earnings": 0.9,  
    "expand": 0.8,  
    "expansion": 1.0,  
    "exceed": 0.9,  
    "excel": 1.0,  
    "exceptional": 1.0,  
    "excellent": 1.0,  
    "enhance": 0.8,  
    "elevate": 0.8,  
    "empower": 1.0,  
    "exponential": 1.0,  
    "efficient": 0.9,  
    "equity": 0.8,  
    "euphoria": 1.0,  
    "entrepreneurial": 0.9,  
    "elite": 0.9,  
    "enrich": 0.8,  
    "exceptionally": 1.2,  
    "elevated": 0.9,  
    "engaged": 0.8,  
    "exhilarating": 1.0,  
    "economiccrisis": -1.0,  
    "end": -0.8,  
    "exhausted": -0.9,  
    "error": -0.8,  
    "exaggerate": -0.7,  
    "embarrassing": -0.9,  
    "expose": -0.8,  
    "exploitation": -1.0,  
    "excessive": -0.7,  
    "evade": -0.8,  
    "evasion": -0.9,  
    "eliminate": -0.6,  
    "embittered": -1.0,  
    "excruciating": -1.0,  
    "enfeeble": -0.8,  
    "exhausting": -0.9,  
    "exclusion": -0.8,  
    "endangered": -0.8,  
    "erosion": -0.9,  
    "exceptionally": 1.2,  # Intensificatore positivo, denota qualcosa di eccezionale
    "efficiently": 0.9,  # Avverbio positivo, denota ottimizzazione e prestazioni elevate
    "exponentially": 1.3,  # Intensificatore positivo, denota crescita rapida e significativa
    "excessively": -0.7,  # Avverbio negativo che denota esagerazione o qualcosa al di là del necessario
    "exhaustively": -0.8,  # Avverbio negativo, denota qualcosa di troppo stancante o estenuante
    "embarrassingly": -1.0,  # Intensificatore negativo, denota qualcosa di imbarazzante
    "evidently": 0.7,  # Avverbio positivo che denota chiarezza o evidenza
    "eagerly": 1.1,  # Avverbio positivo che denota un'aspettativa positiva
    "exclusively": -0.5,  # Avverbio che limita qualcosa in modo negativo, denota esclusività negativa

    "flourish": 1.0,  
    "flourishing": 1.0,  
    "favorable": 0.9,  
    "fantastic": 1.0,  
    "forward": 0.8,  
    "flexible": 0.8,  
    "favorableoutlook": 0.9,  
    "fortuitous": 1.0,  
    "financiallyhealthy": 1.0,  
    "fortune": 1.0,  
    "futureproof": 1.0,  
    "foundational": 0.8,  
    "focus": 0.8,  
    "fame": 0.7,  
    "fasttrack": 1.0,  
    "futuristic": 0.9,  
    "fair": 0.8,  
    "failure": -1.0,  
    "failing": -1.0,  
    "fragile": -0.9,  
    "flawed": -0.8,  
    "falter": -0.9,  
    "fall": -1.0,  
    "frustration": -0.9,  
    "foreclosure": -1.0,  
    "fiasco": -1.0,  
    "famine": -1.0,  
    "fraud": -1.0,  
    "financialcrisis": -1.0,  
    "fallout": -0.8,  
    "feeble": -0.7,  
    "failingmarket": -1.0,  
    "flounder": -0.8,  
    "flop": -1.0,  
    "favorably": 1.1,  # Intensificatore positivo che denota un risultato positivo
    "frequently": 0.7,  # Avverbio neutro, denota frequenza ma senza un'intensificazione chiara
    "falteringly": -1.2,  # Intensificatore negativo, denota una debolezza evidente
    "financially": 0.8,  # Relativo alla stabilità finanziaria, positivo ma non intensivo
    "fearfully": -1.0,  # Intensificatore negativo, denota paura o apprensione
    "flamboyantly": 1.0,  # Intensificatore positivo, denota un comportamento audace
    "faintly": -0.5,  # Intensificatore attenuante, denota una lieve presenza di qualcosa negativo
    "frenetically": -0.7,  # Intensificatore negativo, denota agitazione o caos
    "firmly": 1.0,  # Intensificatore positivo, denota fermezza e determinazione
    "frantically": -1.0,  # Intensificatore negativo, denota ansia o disordine

    "growth": 0.9,  
    "growing": 1.0,  
    "great": 1.0,  
    "gain": 0.9,  
    "gains": 0.9,  
    "good": 0.8,  
    "golden": 1.0,  
    "gigantic": 1.0,  
    "generous": 0.8,  
    "grateful": 0.8,  
    "groundbreaking": 1.0,  
    "glowing": 1.0,  
    "glorious": 1.0,  
    "growingstrong": 1.0,  
    "growthrate": 0.9,  
    "genuine": 0.8,  
    "green": 0.7,  
    "gorgeous": 1.0,
    "grim": -0.9,  
    "grinding": -0.8,  
    "grief": -1.0,  
    "gloom": -0.9,  
    "grind": -0.8,  
    "grousing": -0.7,  
    "greed": -1.0,  
    "gutted": -1.0,  
    "grouchy": -0.7,  
    "glum": -0.8,  
    "glitch": -0.7,  
    "gasp": -0.6,  
    "grip": -0.7,  
    "grievance": -0.9,  
    "grouch": -0.8,  
    "guilt": -1.0,  
    "gratefully": 0.8,  # Avverbio positivo che esprime gratitudine
    "graciously": 0.9,  # Intensificatore positivo che indica comportamento generoso
    "gracefully": 1.0,  # Intensificatore positivo che implica eleganza e successo
    "grimly": -1.0,  # Intensificatore negativo che implica pessimismo
    "greedily": -1.0,  # Intensificatore negativo, denota avidità
    "grouchily": -0.8,  # Intensificatore negativo che denota comportamento scontroso
    "glowingly": 1.1,  # Intensificatore positivo che denota successo e felicità
    "grindingly": -0.9,  # Intensificatore negativo che implica una fatica costante
    "grudgingly": -0.8,  # Intensificatore negativo che implica resistenza o mancanza di entusiasmo




    
    "extremely": 1.5,  # Intensificatore forte
    "strongly": 1.5,  # Intensificatore forte
    "highly": 1.5,  # Intensificatore forte
    "severely": 1.5,  # Intensificatore forte
    "significantly": 1.5,  # Intensificatore
    "moderately": 0.5,  # Attenuatore
    "steadily": 1.2,  # Intensificatore moderato
    "somewhat": 0.5,  # Attenuatore
    "slightly": 0.5,  # Attenuatore
    "unexpectedly": 1.0,  # Neutrale (non altera, ma può suggerire un cambiamento inaspettato)
    "eventually": 1.0,  # Neutrale
    "increasingly": 1.2,  # Intensificatore
    "gradually": 1.0,  # Intensificatore (ma più morbido)
    "dramatically": 1.5,  # Intensificatore forte
    "rapidly": 1.5,  # Intensificatore forte
    "sharply": 1.5,  # Intensificatore forte
    "substantially": 1.5,  # Intensificatore forte
    "slightly": 0.5,  # Attenuatore
}

def calculate_sentiment(titles):
    total_sentiment = 0
    total_absolute_sentiment = 0  # Somma assoluta dei punteggi (per la normalizzazione)

    for title in titles:
        print(f"Analyzing: {title}\n")
        
        # Elaboriamo il testo con spaCy
        doc = nlp(title)

        title_sentiment = 0  # Sentiment totale della notizia

        # Analizziamo le parole nel testo
        for token in doc:
            # Verifica se la parola è nel dizionario di sentiment
            if token.text.lower() in sentiment_dict:
                word_sentiment = sentiment_dict[token.text.lower()]
                total_absolute_sentiment += abs(word_sentiment)  # Sommiamo l'assoluto del punteggio
                
                # Stampa la parola e il suo punteggio
                print(f"Word: {token.text} - Sentiment: {word_sentiment}")

                # Se è un nome modificato da un aggettivo, modifichiamo il punteggio
                if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    for child in token.children:
                        if child.pos_ == "ADJ" or child.pos_ == "ADV":
                            adj_or_adv_sentiment = sentiment_dict.get(child.text.lower(), 0)
                            print(f"  - {child.text} modifies {token.text} with sentiment: {adj_or_adv_sentiment}")
                            word_sentiment += adj_or_adv_sentiment
                
                # Gestiamo gli avverbi che intensificano il verbo (come "sharply")
                if token.pos_ == "VERB" and any(child.pos_ == "ADV" for child in token.children):
                    for child in token.children:
                        if child.pos_ == "ADV" and child.text.lower() in sentiment_dict:
                            word_sentiment *= sentiment_dict[child.text.lower()]
                            print(f"  - {child.text} intensifies {token.text} with multiplier")
                
                # Sommiamo il punteggio del token alla notizia
                title_sentiment += word_sentiment

        # Normalizziamo il punteggio per farlo rientrare tra -1 e 1
        if total_absolute_sentiment > 0:
            normalized_sentiment = title_sentiment / total_absolute_sentiment
            # Limitiamo il punteggio tra -1 e 1
            normalized_sentiment = max(-1, min(1, normalized_sentiment))
        else:
            normalized_sentiment = 0  # Nel caso in cui non ci siano parole con punteggio
        
        # Stampiamo il sentiment totale della notizia
        print(f"Overall Sentiment for this title: {normalized_sentiment}\n")
        
        # Restituiamo la notizia e il sentiment finale
        print(f"Final Sentiment: {title} - {normalized_sentiment}\n")
    

# Recuperiamo i titoli delle notizie per "TSLA" (come esempio)
titles = get_stock_news("TSLA")

# Calcoliamo il sentiment per ogni titolo
calculate_sentiment(titles)
