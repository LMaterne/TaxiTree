import numpy as np

from typing import List

from TaxiTree.node import Node
from .node import Node, NodeType, RandomNode


class Tree:
    def __init__(self, x, y, node_kind: Node, id_: int, energy: int):
        self._nodes = [node_kind(x, y, NodeType.BRANCH)]
        self._age = 0
        self._energy = energy
        self._id = id_

    def update(self, availability) -> List[List[bool]]:
        self._age += 1
        new_nodes = []
        for node in reversed(self._nodes):
            self._energy, new_node = node.update(
                (availability == self._id) | (availability == -1),
                self._age,
                self._energy
            )
            new_nodes.append(new_node)
            if new_node:
                x, y = new_node._position
                x_lim, y_lim = availability.shape
                if new_node._type == NodeType.BRANCH:
                    availability[max(x-2, 0): min(x+3, x_lim), y] = np.where(
                        availability[max(x-2, 0): min(x+3, x_lim), y] == -1,
                        self._id,
                        availability[max(x-2, 0): min(x+3, x_lim), y]
                    )
                    availability[x, max(y-2, 0): min(y_lim, y+3)] = np.where(
                        availability[x, max(y-2, 0): min(y_lim, y+3)] == -1,
                        self._id,
                        availability[x, max(y-2, 0): min(y_lim, y+3)]
                    )
                availability[x, y] = -2
        self._nodes += list(filter(lambda x: x, new_nodes))
        return availability

    def draw(self, axes):
        for node in reversed(self._nodes):
            node.draw(axes)


class RandomTree(Tree):
    def __init__(self, x, y, id_: int, energy: int = 5):
        super().__init__(x, y, RandomNode, id_,  energy)
