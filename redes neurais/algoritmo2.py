import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# BASE DE DADOS
frases = [
    "eu gostei muito disso", "isso é incrível", "estou feliz", "muito bom",
    "excelente trabalho", "eu adorei", "maravilhoso", "fantástico",
    "amei demais", "foi muito bom", "sensacional", "legal demais",
    "muito bom mesmo", "perfeito", "ótimo", "bom trabalho",
    "gostei bastante", "ficou incrível", "isso foi ótimo",
    "curti muito", "show de bola", "top demais", "muito bom isso",
    "foi excelente", "nota 10", "recomendo muito", "adorei isso",
 # negativos
    "eu odiei isso", "isso é ruim", "estou triste", "muito ruim",
    "péssimo", "horrível", "detestei", "muito decepcionante",
    "não gostei", "terrível", "foi ruim", "odiei muito",
    "muito ruim mesmo", "não vale a pena", "pior coisa",
    "decepcionante", "não recomendo", "experiência ruim",
    "isso foi horrível", "não gostei disso", "muito fraco",
    "bem ruim", "não curti", "péssima experiência", "lixo",

 # neutros
    "ok", "normal", "mais ou menos", "tanto faz",
    "não sei", "pode ser", "é aceitável",
    "nada demais", "regular", "mediano"
]
rotulos = (
    ["positivo"] * 27 +
    ["negativo"] * 25 +
    ["neutro"] * 10
)
# TREINO
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(frases)

modelo = SVC(kernel="linear", probability=True)
modelo.fit(X, rotulos)

# ACURÁCIA DO MODELO
scores = cross_val_score(modelo, X, rotulos, cv=2)
acuracia = np.mean(scores)
# NORMALIZAÇÃO
def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r"\bvc\b", "você", texto)
    return texto
# PREVISÃO
def prever(frase):
    frase_original = frase
    frase = normalizar(frase)
    X_input = vectorizer.transform([frase])
    pred = modelo.predict(X_input)[0]
    probs = modelo.predict_proba(X_input)[0]
    confianca = max(probs) * 100
    print("Frase:", frase_original)
    print("Resultado:", pred)
    print(f"Confiança: {confianca:.1f}%")

    print("---------------------")

# TESTES
prever("eu gostei muito disso")
prever("isso é horrível")
prever("ok")
print(f"Acurácia do modelo: {acuracia*100:.1f}%")
