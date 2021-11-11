from lightSOM.som_net import KSOM

class SOM():
    def __init__(self, backend="classic"):
        self.backend=backend


    def create(self, width, height, data, decay_function="exponential_decay", normalizer="var", distance="euclidean", neighborhood="gaussian", lattice="hexa", features_names=[], index=None, target=None,loadFile=None, pci=0, pbc=0, random_seed=None):
        if self.backend=='classic':
            return KSOM(width=width, height=height, data=data, decay_function=decay_function,  normalizer=normalizer, distance=distance, neighborhood=neighborhood, lattice=lattice, features_names=features_names, index=index, target=target, loadFile=loadFile, PCI=pci, PBC=pbc, random_seed=random_seed)