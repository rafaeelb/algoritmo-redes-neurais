import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import re

# ----------------------------
# 1. BASE DE DADOS MAIOR
# ----------------------------
frases = [
    # positivos
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
    # negativos com palavrão
    "isso é uma merda",
    "que lixo isso",
    "horrível pra caralho",
    "muito ruim essa porra",
    "odiei essa merda",
    "isso tá uma bosta",
    "que coisa horrível cara",
    "péssimo pra caramba",

    # neutros
    "ok", "normal", "mais ou menos", "tanto faz",
    "não sei", "pode ser", "é aceitável",
    "nada demais", "regular", "mediano"
]

rotulos = (
    ["positivo"] * 27 +
    ["negativo"] * 33 +
    ["neutro"] * 10
)

# ----------------------------
# 2. TREINO
# ----------------------------
# vetor melhor (pega combinações de palavras)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(frases)

modelo = SVC(kernel='linear', probability=True)
modelo.fit(X, rotulos)

# acurácia real
scores = cross_val_score(modelo, X, rotulos, cv=5)
acuracia = np.mean(scores)

# ----------------------------
# 3. INTERFACE
# ----------------------------
def normalizar_texto(texto):
    texto = texto.lower()

    substituicoes = {
        r"\bvc\b": "você",
        r"\bvcs\b": "vocês",
        r"\btb\b": "também",
        r"\btbm\b": "também",
        r"\bpq\b": "porque",
        r"\bq\b": "que",
        r"\bmt\b": "muito",
        r"\bmto\b": "muito",
        r"\btop\b": "muito bom",
        r"\bkkk+\b": "feliz",
        r"\brs+\b": "feliz",
        r"\baff\b": "ruim",
        r"\bmds\b": "surpresa",
        r"\bn\b": "não",
        r"\bnao\b": "não",
        r"\bmerda\b": "ruim",
        r"\bbosta\b": "ruim",
        r"\bporra\b": "ruim",
        r"\bcaralho\b": "intensidade",
        r"\blixo\b": "ruim"
    }

    for padrao, correto in substituicoes.items():
        texto = re.sub(padrao, correto, texto)

    return texto

class App:
    def __init__(self, root):
        root.title("Análise de Sentimento com IA")
        root.geometry("420x340")
        root.configure(bg="#1e1e1e")

        tk.Label(root, text="Digite uma frase:", bg="#1e1e1e",
                 fg="white", font=("Arial", 12)).pack(pady=10)

        self.entry = tk.Entry(root, width=40, font=("Arial", 12))
        self.entry.pack(pady=5)

        tk.Button(root, text="Analisar", command=self.prever,
                  bg="#4CAF50", fg="white").pack(pady=10)

        self.resultado = tk.Label(root, text="", font=("Arial", 18, "bold"),
                                  bg="#1e1e1e")
        self.resultado.pack(pady=10)

        self.conf = tk.Label(root, text="", font=("Arial", 11),
                             bg="#1e1e1e", fg="gray")
        self.conf.pack()

        self.acc = tk.Label(root,
                            text=f"Acurácia do modelo: {acuracia*100:.1f}%",
                            font=("Arial", 10),
                            bg="#1e1e1e",
                            fg="lightblue")
        self.acc.pack(side="bottom", pady=10)

    def prever(self):
        texto = normalizar_texto(self.entry.get())

        if texto.strip() == "":
            self.resultado.config(text="Digite algo!")
            return

        X_input = vectorizer.transform([texto])

        pred = modelo.predict(X_input)[0]
        probs = modelo.predict_proba(X_input)[0]
        confianca = max(probs) * 100

        if pred == "positivo":
            self.resultado.config(text="😊 Frase positiva", fg="#00ff88")
        elif pred == "negativo":
            self.resultado.config(text="😡 Frase negativa", fg="#ff4c4c")
        else:
            self.resultado.config(text="😐 Frase neutra", fg="#cccccc")

        self.conf.config(text=f"Confiança: {confianca:.1f}%")

# ----------------------------
# 4. RODAR
# ----------------------------
root = tk.Tk()
app = App(root)
root.mainloop()