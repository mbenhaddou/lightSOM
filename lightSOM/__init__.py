from lightSOM.som_net import KSOM

class SOM():
    def __init__(self, backend="classic"):
        self.backend=backend


    def create(self, width, height, data,normalizer="var", distance="euclidean", neighborhood="gaussian", lattice="hexa", feature_names=[], target=None,loadFile=None, pci=0, pbc=0, random_seed=None):
        if self.backend=='classic':
            return KSOM(width=width, height=height, data=data,  normalizer=normalizer, distance=distance, neighborhood=neighborhood, lattice=lattice, features_names=feature_names, target=target, loadFile=loadFile, PCI=pci, PBC=pbc, random_seed=random_seed)