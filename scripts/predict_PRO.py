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
    "extremely": 1.5,  # Intensificatore forte
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

    "high": 0.8,  
    "higher": 0.9,  
    "highs": 0.8,  
    "healthy": 0.9,  
    "hedge": 0.6,  
    "hero": 1.0,  
    "historic": 0.8,  
    "hope": 0.7,  
    "hopeful": 0.9,  
    "hot": 0.7,  
    "huge": 1.0,  
    "hike": 0.6,  
    "headway": 0.7,  
    "hefty": 0.8,  
    "helpful": 0.7,  
    "handsome": 0.9,  
    "harvest": 0.8,  
    "highlight": 0.7,  
    "honeymoon": 0.9,  
    "hypergrowth": 1.0,  
    "hit": -0.8,  
    "halt": -0.9,  
    "hardship": -1.0,  
    "harsh": -0.8,  
    "hazard": -0.9,  
    "heavy": -0.7,  
    "headwind": -0.8,  
    "hedgingloss": -0.9,  
    "hurdle": -0.7,  
    "hurt": -0.9,  
    "hostile": -1.0,  
    "hoax": -1.0,  
    "havoc": -1.0,  
    "hopeless": -1.0,  
    "hollow": -0.8,  
    "halted": -0.9,  
    "hype" : -0.2,  # A volte può essere positivo, ma spesso ha una connotazione negativa  
    "hiddenrisk": -0.9,  
    "heavily": -0.8,  # Intensificatore negativo, denota impatto forte  
    "hopefully": 0.7,  # Attenuatore positivo, indica aspettative positive  
    "harshly": -1.0,  # Intensificatore negativo  
    "honestly": 0.6,  # Leggero intensificatore positivo  
    "hastily": -0.7,  # Intensificatore negativo, suggerisce azioni avventate  
    "hugely": 1.2,  # Intensificatore positivo, enfatizza la grandezza di un evento  
    "historically": 0.5,  # Neutrale, può dare contesto a una tendenza  
    "highly": 1.1,  # Intensificatore positivo, enfatizza importanza o crescita  

    "increase": 0.8,  
    "increasing": 0.9,  
    "innovation": 1.0,  
    "innovative": 1.0,  
    "improve": 0.9,  
    "improvement": 1.0,  
    "improving": 0.9,  
    "income": 0.8,  
    "insightful": 0.7,  
    "investment": 0.8,  
    "investor": 0.7,  
    "influential": 0.8,  
    "inflow": 0.7,  
    "impressive": 1.0,  
    "intelligent": 0.9,  
    "independent": 0.7,  
    "inspiring": 1.0,  
    "ideal": 0.9,  
    "important": 0.8,  
    "initiative": 0.7,  
    "integrity": 0.8,  
    "incentive": 0.7,  
    "inflationresistant": 0.9,  
    "inflation": -0.9,  
    "insolvent": -1.0,  
    "instability": -0.9,  
    "inefficiency": -1.0,  
    "inefficient": -0.9,  
    "insecurity": -0.8,  
    "insufficient": -0.9,  
    "issue": -0.7,  
    "incapable": -0.8,  
    "incompetent": -1.0,  
    "irrelevant": -0.7,  
    "insignificant": -0.7,  
    "inconvenient": -0.7,  
    "inconsistency": -0.8,  
    "incorrect": -0.9,  
    "irregular": -0.8,  
    "isolated": -0.7,  
    "instability": -0.9,  
    "incident": -0.7,  
    "increasingly": 1.2,  # Intensificatore positivo, denota crescita progressiva  
    "insightfully": 1.1,  # Intensificatore positivo, denota saggezza o visione  
    "impressively": 1.2,  # Intensificatore positivo, enfatizza un risultato eccellente  
    "insufficiently": -1.0,  # Intensificatore negativo, denota mancanza  
    "inefficiently": -1.2,  # Intensificatore negativo, enfatizza l'inefficienza  
    "insecurely": -1.0,  # Intensificatore negativo, denota insicurezza  
    "inconsistently": -1.0,  # Intensificatore negativo, denota mancanza di coerenza  

    "jump": 0.8,  
    "jumping": 0.9,  
    "jolt": 0.7,  
    "jubilant": 1.0,  
    "joy": 1.0,  
    "joyful": 1.0,  
    "juicy": 0.8,  # Spesso usato in ambito finanziario per indicare rendimenti attraenti
    "justify": 0.7,  
    "justified": 0.8,  
    "jittery": -0.8,  
    "jeopardy": -1.0,  
    "jeopardize": -1.0,  
    "jaded": -0.7,  
    "jammed": -0.8,  
    "jerky": -0.7,  
    "junk": -1.0,  
    "junkbond": -1.0,  
    "judgmental": -0.9,  
    "jumpiness": -0.7,  
    "joyfully": 1.2,  # Intensificatore positivo, indica entusiasmo e ottimismo
    "jubilantly": 1.3,  # Intensificatore positivo, enfatizza una forte positività
    "jarringly": -1.2,  # Intensificatore negativo, indica un impatto negativo improvviso
    "jitterily": -1.1,  # Intensificatore negativo, enfatizza insicurezza o instabilità
    "justly": 0.6,  # Attenuante positivo, indica equità e correttezza

    "keen": 0.7,  
    "keenness": 0.8,  
    "key": 0.9,  
    "kickstart": 1.0,  
    "king": 1.0,  
    "knockout": 0.9,  
    "knowhow": 0.8,  
    "knowledgeable": 0.9,  
    "keenly": 0.7,  
    "knock": -0.7,  
    "knockeddown": -0.9,  
    "knockoutblow": -1.0,  
    "kill": -1.0,  
    "killing": -1.0,  
    "kneedeep": -0.8,  
    "knotted": -0.7,  
    "kickback": -0.9,  
    "keptdown": -0.8,  
    "keenly": 1.1,  # Intensificatore positivo, indica forte interesse o entusiasmo  
    "knowingly": 0.6,  # Avverbio neutro-positivo, denota consapevolezza  
    "knockingly": -1.0,  # Intensificatore negativo, indica forte critica o attacco  

    "lead": 0.9,  
    "leading": 1.0,  
    "leadership": 1.0,  
    "leap": 0.8,  
    "leaping": 0.9,  
    "leverage": 0.7,  
    "lucrative": 1.0,  
    "lift": 0.8,  
    "landmark": 0.9,  
    "limitless": 1.0,  
    "legendary": 1.0,  
    "lightweight": 0.7,  
    "longterm": 0.8,  
    "loyal": 0.9,  
    "luxury": 0.9,  
    "loss": -1.0, 
    "losses": -1.0,
    "losing": -1.0,  
    "lost": -0.9,  
    "lag": -0.8,  
    "lagging": -0.9,  
    "liability": -0.8,  
    "lawsuit": -0.9,  
    "layoff": -1.0,  
    "limited": -0.7,  
    "low": -0.8,  
    "lack": -0.9,  
    "lousy": -1.0,  
    "looming": -0.8,  
    "liquidation": -1.0,  
    "leak": -0.7,  
    "largely": 0.6,  # Avverbio positivo che indica un effetto significativo  
    "lightly": 0.7,  # Intensificatore positivo, indica leggerezza e agilità  
    "loudly": -0.6,  # Intensificatore negativo se usato in senso di protesta o allarme  
    "likely": 0.5,  # Avverbio positivo ma attenuato, denota possibilità positiva  
    "loosely": -0.5,  # Attenuante negativo, indica mancanza di precisione  

    "market": 0.9,  
    "momentum": 1.0,  
    "money": 0.8,  
    "magnificent": 1.0,  
    "master": 1.0,  
    "merger": 0.8,  
    "modern": 0.7,  
    "maximized": 1.0,  
    "mature": 0.7,  
    "moneyflow": 0.9,  
    "metamorphosis": 1.0,  
    "magnify": 0.8,  
    "marvelous": 1.0,  
    "milestone": 1.0,  
    "motivated": 0.9,  
    "movement": 0.8,  
    "maximizing": 1.0,  
    "marketshare": 0.8,  
    "meltdown": -1.0,  
    "mismanagement": -1.0,  
    "melancholy": -0.8,  
    "misfortune": -0.9,  
    "misstep": -0.8,  
    "malfunction": -1.0,  
    "mishap": -0.8,  
    "mortgage": -0.7,  
    "manipulation": -1.0,  
    "mediocre": -0.7,  
    "minimized": -0.8,  
    "marketdecline": -1.0,  
    "monetaryloss": -1.0,  
    "moneyloss": -0.9,  
    "mistake": -0.9,  
    "monotony": -0.7,  
    "misaligned": -0.8, 
    "magnificently": 1.2,  # Intensificatore positivo, denota eccellenza
    "miserably": -1.0,  # Intensificatore negativo, denota fallimento o disastro
    "moderately": 0.6,  # Attenuante, indica una valutazione neutra o leggermente positiva
    "mercilessly": -1.2,  # Intensificatore negativo, denota un impatto negativo forte
    "manipulatively": -1.0,  # Intensificatore negativo, legato a pratiche manipolative
    "methodically": 0.7,  # Avverbio positivo, denota precisione e organizzazione
    "motivatingly": 1.1,  # Intensificatore positivo, denota azioni motivanti
    "mildly": 0.5,  # Attenuante positivo, che indica una leggera positività

    "net": 0.8,  
    "new": 0.9,  
    "notable": 1.0,  
    "noble": 1.0,  
    "nurture": 0.8,  
    "navigating": 0.7,  
    "nearrecord": 1.0,  
    "niche": 0.7,  
    "nurturing": 0.8,  
    "nice": 0.7,  
    "nifty": 0.9,  
    "noble": 1.0,  
    "nourishing": 0.8,  
    "noteworthy": 1.0,  
    "newmarket": 1.0,  
    "nextlevel": 1.0,  
    "negative": -1.0,  
    "narrow": -0.8,  
    "nagging": -0.7,  
    "nosedive": -1.0,  
    "noxious": -1.0,  
    "nadir": -1.0,  
    "numb": -0.8,  
    "nonperforming": -1.0,  
    "nonviable": -1.0,  
    "nuisance": -0.8,  
    "nervous": -0.8,  
    "negligible": -0.6,  
    "nightmare": -1.0,  
    "needy": -0.7,  
    "negligent": -0.9,  
    "negatively": -1.0,  # Intensificatore negativo, denota una valutazione peggiorativa
    "nervously": -0.7,  # Intensificatore negativo, denota ansia o incertezza
    "narrowly": -0.6,  # Intensificatore negativo, denota una condizione ristretta
    "notably": 0.8,  # Avverbio che indica qualcosa di degno di nota, generalmente positivo
    "naturally": 0.7,  # Avverbio che indica una progressione naturale, positiva
    "negligibly": -0.5,  # Avverbio che attenua una situazione negativa

    "outperformance": 1.0,  
    "outlook": 0.8,  
    "optimistic": 1.0,  
    "opportunity": 0.9,  
    "outperform": 1.0,  
    "outstanding": 1.0,  
    "outpacing": 0.9,  
    "optimal": 0.9,  
    "ontrack": 1.0,  
    "overflowing": 1.0,  
    "opulent": 1.0,  
    "overachieve": 1.0,  
    "overwhelming": 1.0,  
    "organic": 0.7,  
    "overperforming": 0.9,  
    "overseas": 0.7,  
    "opportunityrich": 1.0,  
    "overload": -0.8,  
    "obstacle": -0.9,  
    "outdated": -0.8,  
    "overextended": -0.7,  
    "overdue": -0.8,  
    "oversaturated": -0.9,  
    "overburdened": -0.8,  
    "outofcontrol": -1.0,  
    "obsolete": -0.9,  
    "overpriced": -0.9,  
    "overwhelmed": -0.9,  
    "outage": -0.8,  
    "overconsumption": -0.7,  
    "overreaction": -0.7,  
    "oversupply": -0.9,  
    "offtrack": -0.9,  
    "optimistically": 1.2,  # Intensificatore positivo, indica ottimismo
    "overwhelmingly": 1.3,  # Intensificatore positivo, denota una forza travolgente
    "overcautiously": -0.6,  # Avverbio negativo, denota eccessiva cautela
    "offensively": -1.0,  # Intensificatore negativo, denota un comportamento problematico
    "obviously": 0.7,  # Avverbio positivo che indica chiarezza
    "overzealously": -0.7,  # Avverbio negativo che indica un comportamento eccessivamente entusiasta
    "outwardly": 0.6,  # Avverbio che indica un'apparenza positiva, ma senza implicazioni forti

    "profit": 1.0,  
    "profits": 0.9,  
    "positive": 0.9,  
    "prosperity": 1.0,  
    "progress": 0.9,  
    "promising": 1.0,  
    "productive": 0.8,  
    "potential": 0.8,  
    "powerful": 1.0,  
    "pioneering": 1.0,  
    "praise": 0.9,  
    "prize": 1.0,  
    "performing": 0.8,  
    "profitable": 1.0,  
    "precious": 0.8,  
    "plentiful": 1.0,  
    "progressive": 0.9,  
    "platform": 0.8,  
    "perfect": 1.0,  
    "partnership": 0.8,  
    "plunge": -1.0,  
    "plummeting": -1.0,  
    "problem": -0.9,  
    "poor": -0.9,  
    "pessimistic": -1.0,  
    "pressure": -0.8,  
    "pitfall": -1.0,  
    "panic": -1.0,  
    "paralysis": -1.0,  
    "peril": -1.0,  
    "punish": -0.9,  
    "precarious": -1.0,  
    "pollution": -0.8,  
    "pathos": -0.9,  
    "problems": -0.8,  
    "poorperformance": -1.0,  
    "pricecut": -0.8,  
    "profitably": 1.2,  # Intensificatore positivo che denota guadagni
    "positively": 1.1,  # Intensificatore positivo che enfatizza una situazione favorevole
    "progressively": 1.0,  # Avverbio che denota un progresso continuo
    "perfectly": 1.2,  # Intensificatore positivo, denota prestazioni impeccabili
    "pessimistically": -1.1,  # Intensificatore negativo, suggerisce una visione molto negativa
    "poorly": -1.0,  # Intensificatore negativo, denota una performance scadente
    "precisely": 0.7,  # Avverbio che denota precisione, generalmente neutro ma positivo in contesti finanziari
    "precariously": -1.2,  # Intensificatore negativo, denota una situazione instabile e rischiosa
    "potentially": 0.8,  # Avverbio che indica possibilità positive, ma non garantite
    "painfully": -1.2,  # Intensificatore negativo, suggerisce dolore o difficoltà gravi

    "quality": 0.9,  
    "quick": 0.8,  
    "quicker": 0.9,  
    "quantum": 1.0,  
    "quintessential": 1.0,  
    "qualified": 0.8,  
    "quantitative": 0.8,  
    "quaint": 0.7,  
    "quiet": 0.6,  
    "quest": 1.0,  
    "qualitygrowth": 1.0,
    "quagmire": -1.0,  
    "quicksand": -1.0,  
    "quiver": -0.8,  
    "quashed": -0.9,  
    "questionable": -0.8,  
    "quandary": -0.9,  
    "quarrel": -0.9,  
    "quash": -0.8,  
    "quitting": -0.9,  
    "quizzical": -0.7,  
    "quickly": 0.9,  # Intensificatore positivo, denota un'azione rapida
    "quietly": 0.6,  # Avverbio che attenua, indica tranquillità o discrezione
    "quizzically": -0.7,  # Intensificatore negativo, denota dubbi o incertezze
    "questionably": -0.8,  # Intensificatore negativo, suggerisce incertezze o sospetti
    "quarrelsomely": -1.0,  # Intensificatore negativo, suggerisce conflitti

    "revenue": 0.9,  
    "record": 1.0,  
    "robust": 0.8,  
    "resilient": 1.0,  
    "rise": 0.9,  
    "rising": 1.0,  
    "reliable": 0.8,  
    "rewarding": 1.0,  
    "refreshing": 0.8,  
    "recovery": 0.9,  
    "renaissance": 1.0,  
    "revolutionary": 1.0,  
    "return": 0.8,  
    "reinvigorated": 1.0,  
    "resurgence": 0.9,  
    "revitalization": 1.0,  
    "rapid": 0.9,  
    "recession": -1.0,  
    "reduction": -0.8,  
    "risk": -0.7,  
    "rot": -1.0,  
    "reliance": -0.6,  
    "reversal": -0.9,  
    "reproach": -0.8,  
    "reprimand": -0.9,  
    "retreat": -0.8,  
    "ruin": -1.0,  
    "regression": -0.9,  
    "reduced": -0.7,  
    "receding": -0.8,  
    "reckless": -0.9,  
    "resentment": -0.8,  
    "riskaverse": -0.7,  
    "rocky": -0.8,  
    "reparations": -0.9,  
    "relinquish": -0.7, 
    "robustly": 1.1,  # Intensificatore positivo
    "rapidly": 1.1,  # Intensificatore positivo, indica un miglioramento rapido
    "reluctantly": -0.6,  # Avverbio che attenua la volontà o decisione positiva
    "reliably": 0.8,  # Avverbio positivo che indica affidabilità
    "recklessly": -1.2,  # Intensificatore negativo, indica una gestione imprudente
    "regrettably": -1.0,  # Intensificatore negativo, denota un rimpianto
    "repeatedly": -0.7,  # Avverbio che indica ripetizione, spesso in un contesto negativo
    "resolutely": 1.0,  # Intensificatore positivo, denota determinazione

    "success": 1.0,  
    "surge": 0.9,  
    "strong": 1.0,  
    "stability": 0.8,  
    "steady": 0.8,  
    "skyrocket": 1.0,  
    "stellar": 1.0,  
    "surpassing": 0.9,  
    "strategic": 0.8,  
    "superior": 1.0,  
    "sustainable": 0.9,  
    "savings": 0.7,  
    "solid": 0.8,  
    "sharp": 0.9,  
    "significant": 0.8,  
    "stimulate": 0.7,  
    "spectacular": 1.0,  
    "savvy": 0.9,  
    "steadygrowth": 1.0,  
    "scale": 0.8,  
    "struggle": -0.9,  
    "slump": -1.0,  
    "slowdown": -0.8,  
    "stagnation": -1.0,  
    "subpar": -0.9,  
    "scam": -1.0,  
    "suffering": -0.9,  
    "shrink": -0.8,  
    "suffocate": -1.0,  
    "stiff": -0.7,  
    "shady": -0.8,  
    "sink": -1.0,  
    "stricken": -0.9,  
    "sluggish": -0.8,  
    "scarcity": -0.7,  
    "soured": -0.9,  
    "squander": -1.0,  
    "sick": -0.8,  
    "significantly": 1.2,  # Intensificatore positivo, denota un cambiamento importante
    "successfully": 1.1,  # Intensificatore positivo, denota il raggiungimento di un obiettivo
    "steadily": 0.8,  # Avverbio positivo che denota una crescita costante
    "sharply": 1.5,  # Intensificatore positivo, denota un cambiamento rapido e significativo
    "slowly": -0.7,  # Intensificatore negativo, denota un progresso lento o una difficoltà
    "stubbornly": -0.5,  # Intensificatore negativo, denota resistenza ai cambiamenti
    "strikingly": 1.0,  # Intensificatore positivo, denota una caratteristica eccezionale
    "significantly": 1.2,  # Intensificatore positivo
    "shadily": -1.0,  # Intensificatore negativo, denota comportamenti ambigui o illeciti
    "sparsely": -0.6,  # Intensificatore negativo, denota una scarsità
    "substantially": 1.5,  # Intensificatore forte
    "slightly": 0.5,  # Attenuatore

    "takeoff": 1.0,  
    "trade": 0.6,
    "top": 1.0,  
    "triumph": 1.0,  
    "thrive": 0.9,  
    "tremendous": 1.0,  
    "tangible": 0.8,  
    "topping": 0.9,  
    "thriving": 1.0,  
    "trust": 0.8,  
    "trailblazing": 1.0,  
    "transformative": 1.0,  
    "trending": 0.8,  
    "talent": 0.9,  
    "testament": 1.0,  
    "turnaround": 1.0,  
    "treasure": 1.0,  
    "target": 0.8,  
    "trendsetting": 1.0,  
    "topnotch": 1.0,  
    "tumble": -0.9,  
    "trouble": -1.0,  
    "turmoil": -1.0,  
    "tarnish": -0.8,  
    "threat": -1.0,  
    "tragic": -1.0,  
    "turbulent": -0.9,  
    "trapped": -0.8,  
    "tiring": -0.6,  
    "tax": -0.7,  
    "trivial": -0.6,  
    "tension": -0.9,  
    "tear": -0.8,  
    "tangled": -0.7,  
    "tough": -0.7,  
    "threatening": -1.0,  
    "triumphantly": 1.2,  # Intensificatore positivo, esprime una vittoria netta
    "tremendously": 1.2,  # Intensificatore positivo, indica grande successo
    "tactically": 0.8,  # Avverbio neutro, che implica una mossa ben ponderata
    "tensely": -0.8,  # Intensificatore negativo, denota una situazione di alta pressione
    "tragically": -1.2,  # Intensificatore negativo, denota una situazione drammatica
    "turbulently": -1.1,  # Intensificatore negativo, denota caos o instabilità
    "tactfully": 0.7,  # Avverbio positivo, che indica una gestione o azione diplomatica
    "timidly": -0.6,  # Intensificatore negativo, denota incertezza o paura

    "up": 0.7,  
    "uptick": 0.8,  
    "upgrade": 0.9,  
    "upgraded": 1.0,  
    "uptrend": 1.0,  
    "upswing": 0.9,  
    "upside": 0.8,  
    "upbeat": 0.9,  
    "unstoppable": 1.0,  
    "unshakable": 0.9,  
    "unprecedented": 1.0,  
    "undervalued": 0.7,  
    "unique": 0.8,  
    "unwavering": 0.9,  
    "unmatched": 1.0,  
    "unlimited": 0.9,  
    "unbeatable": 1.0,  
    "uncertain": -0.8,  
    "uncertainty": -1.0,  
    "unemployment": -1.0,  
    "underperform": -0.9,  
    "underperformed": -1.0,  
    "underperforming": -1.0,  
    "underwhelming": -0.8,  
    "unprofitable": -1.0,  
    "unrecoverable": -1.0,  
    "unrealized": -0.7,  
    "unsuccessful": -1.0,  
    "unstable": -0.9,  
    "unrest": -1.0,  
    "unexpectedly": -0.6,  
    "unsecured": -0.8,  
    "unfavorable": -1.0,  
    "unclear": -0.7,  
    "upward": 0.9,  # Indica un movimento positivo o una crescita
    "unexpectedly": -0.6,  # Indica un evento inaspettato, può essere negativo
    "unusually": -0.5,  # Attenuante che può indicare variabilità
    "unquestionably": 1.1,  # Intensificatore positivo, rafforza un'affermazione
    "unfortunately": -1.0,  # Intensificatore negativo
    "ultimately": 0.6,  # Avverbio che può attenuare o rafforzare un concetto
    "undoubtedly": 1.0,  # Intensificatore positivo, denota certezza
    "urgently": -0.8,  # Spesso usato in contesti negativi di crisi
    "uncertainly": -0.7,  # Indica un'inaspettata situazione spesso negativa

    "value": 0.8,  
    "valuable": 0.9,  
    "valuation": 0.7,  
    "venture": 0.8,  
    "versatile": 0.7,  
    "vibrant": 0.9,  
    "victory": 1.0,  
    "vigorous": 0.8,  
    "visionary": 1.0,  
    "vital": 0.9,  
    "viable": 0.8,  
    "velocity": 0.7,  
    "venturecapital": 0.9,  
    "vault": 0.8,  
    "vindicated": 0.9,  
    "volatile": -0.8,  
    "volatility": -1.0,  
    "vulnerable": -0.9,  
    "vacillate": -0.7,  
    "vanish": -0.8,  
    "void": -1.0,  
    "vexing": -0.9,  
    "violation": -1.0,  
    "victim": -0.8,  
    "veto": -0.7,  
    "vanquished": -1.0,  
    "vastly": 1.1,  # Intensificatore positivo, indica una grandezza significativa  
    "vigorously": 1.2,  # Intensificatore positivo, indica forza e determinazione  
    "visibly": 0.7,  # Avverbio positivo che enfatizza chiarezza  
    "voluntarily": 0.6,  # Avverbio attenuante, indica scelta consapevole  
    "vaguely": -0.6,  # Avverbio attenuante, indica incertezza  
    "violently": -1.2,  # Intensificatore negativo, denota instabilità o impatto forte  
    "vertiginously": -1.0,  # Intensificatore negativo, indica forte caduta o instabilità  

    "wealth": 1.0,  
    "wealthy": 1.0,  
    "wellbeing": 0.9,  
    "winning": 1.0,  
    "win": 0.9,  
    "wisdom": 0.8,  
    "worthwhile": 0.7,  
    "welcome": 0.8,  
    "widespreadgrowth": 0.9,  
    "workable": 0.7,  
    "wellmanaged": 0.9,  
    "worldclass": 1.0,  
    "wonders": 1.0,  
    "whopping": 0.8,  
    "willpower": 0.8,  
    "weakness": -1.0,  
    "weak": -0.9,  
    "worse": -1.0,  
    "worst": -1.0,  
    "worsening": -1.0,  
    "worry": -0.8,  
    "worrisome": -0.9,  
    "waste": -1.0,  
    "waning": -0.8,  
    "withering": -0.9,  
    "withdrawal": -0.8,  
    "woes": -0.9,  
    "wreck": -1.0,  
    "worthless": -1.0,  
    "wobbly": -0.8,  
    "washedout": -0.9, 
    "widely": 0.7,  # Indica ampia diffusione, attenuante positivo  
    "wisely": 1.1,  # Intensificatore positivo, denota scelte sagge  
    "wildly": -0.7,  # Può attenuare negativamente se riferito a volatilità  
    "wrongly": -1.0,  # Intensificatore negativo, indica errore o ingiustizia  
    "weakly": -1.0,  # Intensificatore negativo, rafforza debolezza  
    "woefully": -1.2,  # Intensificatore negativo molto forte  
    "warmly": 0.8,  # Avverbio positivo, indica accoglienza positiva  
    "wonderfully": 1.2,  # Intensificatore positivo molto forte  

    "xenial": 0.7,  # Indica ospitalità e buone relazioni, positivo per partnership
    "x factor": 0.9,  # Espressione che indica un vantaggio speciale o una qualità unica
    "xenodochial": 0.6,  # Indica cordialità e accoglienza, utile in un contesto aziendale
    "xcelerate": 1.0,  # Forma modificata di "accelerate", spesso usata in branding per crescita
    "xtraordinary": 1.0,  # Variante di "extraordinary", sinonimo di successo e innovazione
    "xenophobia": -1.0,  # Termine negativo che indica ostilità verso gli stranieri, dannoso per il business globale
    "xhausted": -0.9,  # Forma modificata di "exhausted", usata per indicare risorse o mercati esauriti
    "xcruciating": -1.0,  # Variante di "excruciating", usata per indicare difficoltà finanziarie estreme
    "xcluded": -0.8,  # Variante di "excluded", negativa per mercati o aziende escluse da opportunità
    "xcessively": -1.0,  # Variante di "excessively", indica esagerazione spesso con connotazione negativa
    "xceptionally": 1.2,  # Variante di "exceptionally", enfatizza un risultato positivo in modo forte
    "xponentially": 1.1,  # Variante di "exponentially", suggerisce crescita rapida e progressiva

    "yield": 0.8,  # Rendimento, spesso positivo in ambito finanziario  
    "yielding": 0.7,  # Produzione di profitti o risultati positivi  
    "yes": 0.9,  # Espressione di conferma positiva  
    "youthful": 0.6,  # Vitalità e innovazione  
    "yearlygrowth": 1.0,  # Crescita annuale, fortemente positiva  
    "yenrally": 0.8,  # Rally della valuta yen, positivo nei mercati  
    "yielddrop": -0.8,  # Diminuzione dei rendimenti, negativo in finanza  
    "yieldcurveinversion": -1.0,  # Segnale di recessione, fortemente negativo  
    "yawn": -0.6,  # Noioso, poco interessante  
    "yoyo": -0.8,  # Alta volatilità, spesso negativa per gli investitori  
    "yenweakness": -0.9,  # Debolezza della valuta yen, negativo nei mercati  
    "yearly": 0.7,  # Indica continuità e stabilità nel tempo  
    "yawningly": -0.9,  # Espressione di noia o stagnazione  
    "youthfully": 0.8,  # Denota energia e innovazione  
    "yieldingly": 0.6,  # Indica flessibilità e adattabilità  

    "zenith": 1.0,  # Il punto più alto, indica massimo successo
    "zest": 0.8,  # Entusiasmo ed energia positiva
    "zesty": 0.7,  # Spirito vivace e dinamico
    "zeal": 0.9,  # Grande determinazione e passione
    "zealous": 0.8,  # Forte dedizione, spesso in un contesto positivo
    "zoom": 0.9,  # Rapida crescita o aumento
    "zippy": 0.7,  # Vivace, energico, attivo
    "zillion": 0.6,  # Termine iperbolico per enormi quantità (usato positivamente)
    "zero": -1.0,  # Assenza di valore, molto negativo in finanza
    "zapped": -0.8,  # Esausto, eliminato, colpito negativamente
    "zealous" : -0.6,  # Può avere una connotazione negativa di eccesso o ossessione
    "zigzag": -0.7,  # Indica instabilità e mancanza di direzione
    "zombie": -1.0,  # Spesso usato per "zombie companies", aziende senza crescita
    "zoneofuncertainty": -0.9,  # Indica incertezza e instabilità
    "zappedout": -0.8,  # Esaurito, senza energia, negativo per aziende o mercati
    "zeroed": -0.9,  # Azzerato, fallito, estremamente negativo
    "zealously": 1.1,  # Intensificatore positivo, denota grande impegno
    "zestfully": 1.2,  # Intensificatore positivo, denota entusiasmo ed energia
    "zigzaggingly": -1.0,  # Intensificatore negativo, denota incertezza e volatilità
    "zeroingly": -1.2,  # Intensificatore negativo, indica azzeramento e fallimento
}


def normalize_text(text):
    """ Pulisce e normalizza il testo per una migliore corrispondenza. """
    text = text.lower()  # Converti tutto in minuscolo
    text = re.sub(r'[-_/]', ' ', text)  # Sostituisci trattini e underscore con spazi
    text = re.sub(r'\s+', ' ', text).strip()  # Rimuovi spazi multipli e spazi iniziali/finali
    return text


def calculate_sentiment(titles):
    for title in titles:
        print(f"Analyzing: {title}\n")
        normalized_title = normalize_text(title)  # Normalizza il titolo
        
        # Elaboriamo il testo con spaCy
        doc = nlp(normalized_title)

        title_sentiment = 0  # Sentiment totale della notizia
        sentiment_values = []  # Lista per memorizzare i punteggi sentiment per la normalizzazione

        # Analizziamo le parole nel testo
        for token in doc:
            lemma = token.lemma_.lower()  # Usiamo il lemma della parola per cercarlo nel dizionario
            
            if lemma in sentiment_dict:  # Controlliamo il lemma invece della forma originale
                word_sentiment = sentiment_dict[lemma]
                sentiment_values.append(word_sentiment)  # Aggiungiamo il sentiment alla lista
                
                # Stampa la parola e il suo punteggio
                print(f"Word: {token.text} (Lemma: {lemma}) - Sentiment: {word_sentiment}")

                # Se è un nome modificato da un aggettivo, modifichiamo il punteggio
                if token.pos_ in {"NOUN", "PROPN"}:
                    for child in token.children:
                        if child.pos_ in {"ADJ", "ADV"}:
                            adj_or_adv_sentiment = sentiment_dict.get(child.lemma_.lower(), 0)
                            print(f"  - {child.text} modifies {token.text} with sentiment: {adj_or_adv_sentiment}")
                            word_sentiment += adj_or_adv_sentiment
                
                # Gestiamo gli avverbi che intensificano il verbo (come "sharply")
                if token.pos_ == "VERB" and any(child.pos_ == "ADV" for child in token.children):
                    for child in token.children:
                        adv_lemma = child.lemma_.lower()
                        if child.pos_ == "ADV" and adv_lemma in sentiment_dict:
                            word_sentiment *= sentiment_dict[adv_lemma]
                            print(f"  - {child.text} intensifies {token.text} with multiplier")
                
                # Sommiamo il punteggio del token alla notizia
                title_sentiment += word_sentiment

        # Calcoliamo il sentiment finale basato solo sulle parole con sentiment
        if sentiment_values:
            normalized_sentiment = sum(sentiment_values) / len(sentiment_values)  # Media del sentiment
            normalized_sentiment = max(-1, min(1, normalized_sentiment))  # Limitiamo tra -1 e 1
        else:
            normalized_sentiment = 0  # Nessuna parola con punteggio
        
        # Stampiamo il sentiment totale della notizia
        #print(f"Overall Sentiment for this title: {normalized_sentiment}\n")
        print(f"Final Sentiment: {title} - {normalized_sentiment}\n")

# Recuperiamo i titoli delle notizie per "TSLA" (come esempio)
titles = get_stock_news("TSLA")

# Calcoliamo il sentiment per ogni titolo
calculate_sentiment(titles)
