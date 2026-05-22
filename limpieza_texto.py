from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import spacy

# Descargar recursos de nltk
nltk.download('punkt')

# Cargar modelo en español
nlp = spacy.load("es_core_news_sm")

# Leer archivo txt
with open("libro.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()


texto = texto.lower()

texto = re.sub(r'\d+', '', texto)

texto = re.sub(r'[^\w\s]', '', texto)

texto = re.sub(r'\s+', ' ', texto).strip()

print("Texto normalizado:\n")
print(texto[:500])


doc = nlp(texto)

lemmas = [token.lemma_ for token in doc if not token.is_stop]

texto_lematizado = " ".join(lemmas)

print("\n\nTexto lematizado:\n")
print(texto_lematizado[:500])


with open("texto_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_lematizado)

print("\nProceso terminado correctamente.")
