import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

credit = pd.read_csv("Credit.csv", header=0)  
credit.head()



credits = credit.drop(columns=['Unnamed: 0'])
credits.head()

train = credits.sample(frac = 0.5)
data = credits.drop(train.index)
dev = data.sample(frac = 0.5)
eval_set = data.drop(dev.index)

print(train.shape)
print(dev.shape)
print(eval_set.shape)


plt.matshow(train.corr())
plt.xticks(range(len(train.corr().columns)), train.corr().columns);
plt.yticks(range(len(train.corr().columns)), train.corr().columns);

import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

