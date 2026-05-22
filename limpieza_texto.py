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

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


tokens = texto_lematizado.split()

sentencias = [tokens]

modelo_w2v = Word2Vec(
    sentencias,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

print("\nModelo Word2Vec entrenado correctamente.")



palabras = list(modelo_w2v.wv.index_to_key)[:20]

vectores = [modelo_w2v.wv[palabra] for palabra in palabras]



pca = PCA(n_components=2)

resultado = pca.fit_transform(vectores)



plt.figure(figsize=(10,8))

for i, palabra in enumerate(palabras):
    plt.scatter(resultado[i,0], resultado[i,1])
    plt.text(resultado[i,0]+0.01, resultado[i,1]+0.01, palabra)

plt.title("Espacio Vectorial Word2Vec")
plt.savefig("grafica_word2vec.png")

print("\nImagen grafica_word2vec.png guardada.")




plt.figure(figsize=(10,8))

x = resultado[:,0]
y = resultado[:,1]

plt.plot(x, y, 'o')

for i, palabra in enumerate(palabras):
    plt.text(x[i]+0.01, y[i]+0.01, palabra)

plt.title("Representación Semántica del Texto")
plt.savefig("grafica_semantica.png")

print("Imagen grafica_semantica.png guardada.")