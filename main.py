import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sompy.sompy import SOMFactory
from sompy.visualization.umatrix import UMatrixView

import sompy

np.random.seed(0)

# get part of the dataset
# train = pd.read_csv('/Users/mohamedmentis/Dropbox/My Mac (MacBook-Pro.local)/Documents/Mentis/Development/NGA/Projects/Calc/Notebooks/minst/train.csv')
# train = train.sample(n=600, random_state=0)
# labels = train['label']
# train = train.drop("label",axis=1)

# check distribution
#sns.countplot(labels)

# standardization of a dataset
#train_st = StandardScaler().fit_transform(train.values)
raw_data =np.asarray([[1, 0, 0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.2,0.2,0.5]])
names=['r','g', 'b']

from sompy.sompy import SOMFactory
sm = SOMFactory().build(raw_data, (20,20), normalization = 'None', initialization='random', component_names=['r', 'g', 'b'], lattice="hexa")

sm.train(n_job=4, verbose=False, train_rough_len=30, train_finetune_len=100)

hits  = UMatrixView(20, 20,"Clustering",text_size=13)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7)


