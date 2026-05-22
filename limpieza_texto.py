from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import spacy


nltk.download('punkt')

nlp = spacy.load("es_core_news_sm")
nlp.max_length = 3000000

with open("libro.txt", "r", encoding="latin-1") as archivo:
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

# Guardar texto limpio
with open("texto_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_lematizado)


documentos = [texto_lematizado]

vectorizer_count = CountVectorizer()

X_count = vectorizer_count.fit_transform(documentos)

print("\nVectorización CountVectorizer:\n")
print(X_count.toarray())

print("\nPalabras detectadas:\n")
print(vectorizer_count.get_feature_names_out())


vectorizer_tfidf = TfidfVectorizer()

X_tfidf = vectorizer_tfidf.fit_transform(documentos)

print("\nVectorización TF-IDF:\n")
print(X_tfidf.toarray())

print("\nPalabras TF-IDF:\n")
print(vectorizer_tfidf.get_feature_names_out())

print("\nProceso terminado correctamente.")
