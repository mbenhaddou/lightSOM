from lightSOM import SOM
import pandas as pd

democracy_index = pd.read_csv('examples/data/democracy_index.csv')

feature_names = ['democracy_index', 'electoral_processand_pluralism', 'functioning_of_government',
                 'political_participation', 'political_culture', 'civil_liberties']

X = democracy_index[feature_names].values
target=democracy_index['category'].values
names=democracy_index['country_code'].values

size = 15
som=SOM().create(10, 20, X,lattice='rect', features_names=feature_names, target=target, index=names, pci=True, pbc=False)


som.train(start_learning_rate=1,epochs=10000, verbose=True)



from lightSOM.visualization.som_view import SOMView
vhts  = SOMView(som, 10,10, text_size=10)
vhts.plot_features_map()