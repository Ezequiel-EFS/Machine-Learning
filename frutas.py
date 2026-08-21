#%%


import pandas as pd
df = pd.read_excel("data/dados_frutas.xlsx")
print(df) 
# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier()
#%%

y = df["Fruta"]

caracteristicas = ["Arredondada",  "Suculenta",  "Vermelha", "Doce"]
x = df[caracteristicas]

#%%
x

#%%
y
# %%
arvore.fit(x,y)

#%%
arvore.predict([[1,0,1,0]])
# %%

import matplotlib.pyplot as plt
plt.figure(dpi=500)

tree.plot_tree(arvore,feature_names=caracteristicas,
               class_names=arvore.classes_
               ,filled=True)

# %%

proba = arvore.predict_proba([[1,1,1,1]])[0]

pd.Series(proba,index=arvore.classes_)

# %%
