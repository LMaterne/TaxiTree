import numpy as np

from collections import namedtuple
from enum import Enum
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from random import random
from typing import Tuple, List


RADIUS = 0.7
REL_SHIFT =  np.array([1, 0])


NodeDefinition = namedtuple("Node", ["color", "cost", "return_"])


class NodeType(Enum):
    BRANCH = NodeDefinition(color="tab:brown", cost=2, return_=0)
    LEAF = NodeDefinition(color="tab:green", cost=1, return_=1)


class Node:
    def __init__(
            self,
            x: int,
            y: int,
            type_: NodeType,
            previous: "Node"= None
        ):
        self._position = (x, y)
        self._type = type_
        self._prev = previous

    def update(
        self,
        availability: List[List[bool]],
        age: int,
        energy: int
    ) -> Tuple[int, "Node"]:
        consumption, node = self.decision(availability, age, energy)
        consumption += self._type.value.return_
        return energy + consumption, node

    def decision(
            self, availability: List[List[bool]], age: int, energy: int
        ) -> Tuple[int, "Node"]:
        raise NotImplementedError()

    def draw(self, axes: Axes):
        if self._type == NodeType.LEAF:
            axes.add_artist(
                Circle(self._position, radius=RADIUS, color=self._type.value.color)
            )
        else:
            if self._prev:
                if self._position[1] - self._prev._position[1] != 0:
                    height = 1
                    width = 0.7
                    x_off = width / 2
                    y_off = 0.5
                else:
                    width = 1.3
                    height = 0.7
                    x_off = width/2
                    y_off = 0.2
            else:
                height = 1
                width = 0.7
                x_off = width / 2
                y_off = 0.5
            axes.add_artist(
                Rectangle(
                    (self._position[0] - x_off, self._position[1] - y_off),
                    width=width, height=height, color=self._type.value.color
                )
            )

    def _compute_available(self, availability):
        x_max, y_max = availability.shape
        available_spaces = np.zeros((3, 3), dtype=np.bool_)
        if self._type == NodeType.LEAF:
            return available_spaces
        if self._position[0] - 1 >= 0:
            available_spaces[0, 0] = availability[
                self._position[0] - 1, self._position[1]
            ]
        if self._position[0] + 1 < x_max:
            available_spaces[2, 0] = availability[
                self._position[0] + 1, self._position[1]
            ]
        if self._position[1] + 1 < y_max:
            available_spaces[1, 1] = availability[
                self._position[0], self._position[1] + 1
            ]
        return available_spaces

    @classmethod
    def new_leaf(cls, x: int, y: int, type_: NodeType, parent: "Node"):
        return cls(x, y, type_, parent)


class RandomNode(Node):
    def decision(
            self, availability: List[List[bool]], age: int, energy: int
        ) -> Tuple[int, Node]:
        _, y_max = availability.shape
        new_node = NodeType.BRANCH if random() > self._position[1] / y_max else NodeType.LEAF
        available_space = self._compute_available(availability)
        if energy >= new_node.value.cost and available_space.any():
            position = self._generate_new_position(available_space)
            if position:
                return (
                    -new_node.value.cost,
                    RandomNode.new_leaf(*position, new_node, self)
                )
        return 0, None

    def _generate_new_position(self, availability) -> Tuple[int, int]:
        offsets = np.transpose(np.nonzero(availability)) - REL_SHIFT
        p = 1/(self._position[1]+1)**2*np.where(offsets[:, 1] != 0, 1, self._position[1])
        if p.sum() == 0:
            return None
        idx = np.random.choice(range(len(offsets)), p=p/p.sum())
        x_off, y_off = offsets[idx]
        return self._position[0] + x_off, self._position[1] + y_off
