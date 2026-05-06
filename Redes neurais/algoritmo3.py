import tkinter as tk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# BASE DE DADOS
frases = [
# positivos
"eu gostei muito disso", "isso é incrível", "estou feliz", "muito bom",
"excelente trabalho", "eu adorei", "maravilhoso", "fantástico",
"amei demais", "sensacional", "perfeito", "ótimo", "gostei bastante",
"ficou incrível", "curti muito", "foi excelente", "recomendo muito",
"superou minhas expectativas", "muito bem feito", "adorei o resultado",
"que experiência incrível", "estou muito satisfeito", "valeu muito a pena",
"fiquei impressionado", "simplesmente perfeito", "melhor do que esperava",
"que coisa boa", "ficou lindo", "gostei demais", "parabéns mesmo",
"ficou show", "nota 10", "bom trabalho", "adorei cada momento", "aprovado",

# negativos
"eu odiei isso", "isso é ruim", "muito ruim", "péssimo", "horrível",
"detestei", "não gostei", "terrível", "não vale a pena", "decepcionante",
"não recomendo", "experiência ruim", "muito fraco", "não curti",
"péssima experiência", "perda de tempo", "fiquei decepcionado",
"deixou a desejar", "nunca mais", "muito mal feito",
"fiquei muito insatisfeito", "não funcionou", "pior do que esperava",
"me decepcionou muito", "muito abaixo do esperado", "uma decepção",
"não recomendo de jeito nenhum", "terrível mesmo", "bem ruim",
"isso foi horrível", "não gostei disso", "odiei muito",
"péssimo atendimento", "muito ruim mesmo", "que experiência ruim",

# neutros
"ok", "normal", "mais ou menos", "tanto faz", "não sei",
"é aceitável", "nada demais", "regular", "mediano", "razoável",
"nem bom nem ruim", "depende", "talvez", "sei lá", "indiferente",
"é o que é", "pode ser", "não tenho opinião", "mais ou menos isso",
"poderia ser melhor"
]

rotulos = ["positivo"] * 35 + ["negativo"] * 35 + ["neutro"] * 20

# NORMALIZAÇÃO
def normalizar(texto):
    texto = texto.lower()
    subs = {
        r"\bvc\b": "você", r"\bmt\b": "muito", r"\bmto\b": "muito",
        r"\btop\b": "muito bom", r"\bkkk+\b": "feliz", r"\brs+\b": "feliz",
        r"\baff\b": "ruim", r"\bn\b": "não", r"\bnao\b": "não",
    }
    for p, c in subs.items():
        texto = re.sub(p, c, texto)
    return texto

# VETORIZAÇÃO
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform([normalizar(f) for f in frases]).toarray()

# CONVERTER RÓTULOS
encoder = LabelEncoder()
y = encoder.fit_transform(rotulos)

# DIVISÃO TREINO/TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# REDE NEURAL
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# TREINAMENTO
model.fit(X_train, y_train, epochs=50, verbose=0)

# AVALIAÇÃO
loss, acuracia = model.evaluate(X_test, y_test, verbose=0)

# INTERFACE
class App:
    def __init__(self, root):
        root.title("Análise de Sentimento com RNA")
        root.geometry("420x340")
        root.configure(bg="#1e1e1e")

        tk.Label(root, text="Digite uma frase:", bg="#1e1e1e",
                 fg="white", font=("Arial", 12)).pack(pady=10)

        self.entry = tk.Entry(root, width=40, font=("Arial", 12))
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", lambda e: self.prever())

        tk.Button(root, text="Analisar", command=self.prever,
                  bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        self.resultado = tk.Label(root, text="", font=("Arial", 18, "bold"),
                                 bg="#1e1e1e")
        self.resultado.pack(pady=10)

        self.conf = tk.Label(root, text="", font=("Arial", 11),
                             bg="#1e1e1e", fg="gray")
        self.conf.pack()

        tk.Label(root, text=f"Acurácia: {acuracia*100:.1f}%",
                 font=("Arial", 10), bg="#1e1e1e", fg="lightblue"
                 ).pack(side="bottom", pady=10)

    def prever(self):
        texto = normalizar(self.entry.get())

        if not texto.strip():
            self.resultado.config(text="Digite algo!", fg="white")
            return

        X_input = vectorizer.transform([texto]).toarray()
        probs = model.predict(X_input)[0]

        pred = np.argmax(probs)
        confianca = np.max(probs) * 100

        classe = encoder.inverse_transform([pred])[0]

        cores = {
            "positivo": ("#00ff88", "Frase positiva"),
            "negativo": ("#ff4c4c", "Frase negativa"),
            "neutro": ("#cccccc", "Frase neutra")
        }

        self.resultado.config(text=cores[classe][1], fg=cores[classe][0])
        self.conf.config(text=f"Confiança: {confianca:.1f}%")

root = tk.Tk()
App(root)
root.mainloop()