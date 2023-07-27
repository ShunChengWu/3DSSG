import numpy as np


class Node(object):
    def __init__(self, center: np.array, dimension: np.array, neighbors: np.array,
                 Rwc: np.array,
                 kfs: dict,
                 gtInstance: dict = None,
                 label: str = str()):
        super().__init__()
        self.center = center
        self.dimension = dimension
        self.kfs = kfs
        self.neighbors = neighbors
        self.Rwc = Rwc
        self.label = label
        self.gtInstance = gtInstance

    def __repr__(self):
        # print('hello')
        ss = 'center: {}, dimension: {}. kfs: {}'.format(
            self.center, self.dimension, self.kfs)
        return ss

    def __str__(self):
        ss = 'center: {}, dimension: {}. kfs: {}'.format(
            self.center, self.dimension, self.kfs)
        return ss

    def size(self):
        return self.dimension[2]*self.dimension[1]*self.dimension[0]
