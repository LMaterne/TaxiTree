import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple

from .tree import Tree


class World:
    def __init__(
            self,
            trees: List[Tree],
            domain: Tuple[int, int],
        ):
        self._trees = trees
        self._xlim, self._ylim = domain

        self._availability = np.zeros(domain, dtype=np.int32) - 1
        for tree in trees:
            x, y = tree._nodes[0]._position
            self._availability[max(x-2, 0): min(x+3, self._xlim), y] = tree._id
            self._availability[x, y:y+3] = tree._id
            self._availability[x, y] = -2

    def update(self):
        np.random.shuffle(self._trees)
        for tree in self._trees:
            self._availability = tree.update(self._availability)

    def draw(self):
        fig, ax = plt.subplots(
            1, 1, figsize=(2 * self._xlim, 2 * self._ylim))
        for tree in self._trees:
            tree.draw(ax)
        ax.imshow(self._availability.T, aspect="auto", cmap="gray_r", vmin=-2)
        ax.set_xlim(0, self._xlim)
        ax.set_ylim(0, self._ylim)
        ax.axis("off")
        plt.show()
