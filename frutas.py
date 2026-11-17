from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# Caminho da base de dados (funciona de qualquer pasta onde o script for executado)
PASTA = Path(__file__).parent
ARQUIVO = PASTA / "data" / "dados_frutas.xlsx"

CARACTERISTICAS = ["Arredondada", "Suculenta", "Vermelha", "Doce"]


def carregar_dados():
    df = pd.read_excel(ARQUIVO)
    x = df[CARACTERISTICAS]
    y = df["Fruta"]
    return df, x, y


def treinar_arvore(x, y):
    arvore = tree.DecisionTreeClassifier(random_state=42)
    arvore.fit(x, y)
    return arvore


def prever(arvore, valores):
    """Recebe uma lista [Arredondada, Suculenta, Vermelha, Doce] com 0 ou 1."""
    entrada = pd.DataFrame([valores], columns=CARACTERISTICAS)
    fruta = arvore.predict(entrada)[0]
    probabilidades = pd.Series(arvore.predict_proba(entrada)[0], index=arvore.classes_)
    return fruta, probabilidades


def plotar_arvore(arvore, salvar_em=None):
    plt.figure(figsize=(12, 8), dpi=100)
    tree.plot_tree(
        arvore,
        feature_names=CARACTERISTICAS,
        class_names=arvore.classes_,
        filled=True,
    )
    if salvar_em:
        plt.savefig(salvar_em, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    df, x, y = carregar_dados()
    print("=== Dados ===")
    print(df, end="\n\n")

    arvore = treinar_arvore(x, y)

    print("=== Previsões ===")
    for valores in ([1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]):
        fruta, probabilidades = prever(arvore, valores)
        print(f"Características {valores} -> {fruta}")
        print(probabilidades.to_string(), end="\n\n")

    plotar_arvore(arvore)


if __name__ == "__main__":
    main()