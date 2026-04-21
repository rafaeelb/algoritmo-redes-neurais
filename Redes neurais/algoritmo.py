import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import re

# ----------------------------
# 1. BASE DE DADOS (REDUZIDA)
# ----------------------------
frases = [
    # positivos
    "eu gostei muito disso", "isso é incrível", "muito bom",
    "excelente trabalho", "eu adorei", "maravilhoso",
    "fantástico", "ótimo", "bom trabalho", "nota 10",

    # negativos
    "eu odiei isso", "isso é ruim", "muito ruim",
    "péssimo", "horrível", "não gostei", "terrível",
    "decepcionante", "experiência ruim", "lixo",

    # neutros
    "ok", "normal", "mais ou menos", "tanto faz",
    "não sei", "é aceitável", "regular", "mediano"
]

rotulos = (
    ["positivo"] * 10 +
    ["negativo"] * 10 +
    ["neutro"] * 8
)

# ----------------------------
# 2. TREINO
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(frases)

modelo = SVC(kernel='linear', probability=True)
modelo.fit(X, rotulos)

scores = cross_val_score(modelo, X, rotulos, cv=5)
acuracia = np.mean(scores)

# ----------------------------
# 3. INTERFACE
# ----------------------------
def normalizar_texto(texto):
    texto = texto.lower()

    substituicoes = {
        r"\bvc\b": "você",
        r"\bmt\b": "muito",
        r"\blixo\b": "ruim",
        r"\bmerda\b": "ruim",
        r"\bbosta\b": "ruim",
        r"\bporra\b": "ruim",
        r"\bcaralho\b": "intensidade"
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