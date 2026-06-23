import numpy as np
import random
import matplotlib.pyplot as plt

# Tree node
class Node:
    def __init__(self, address, parent=None):
        self.address = address
        self.parent = parent
        self.children = {}
        self.visit_counts = 0
        self.total_value = 0

# MCTS Implementation
class MCTS:
    def __init__(self, max_depth, C, leaf_values, max_iterations_per_root, num_rollouts):
        self.max_iterations_per_root = max_iterations_per_root
        self.max_depth = max_depth
        self.C = C
        self.leaf_values = leaf_values
        self.num_rollouts = num_rollouts
        self.root = Node("")
        self.rng = random.Random(42)
        self.visited_nodes = set()

    # Get UCB score
    def UCB_score(self, child, parent):
        if child.visit_counts == 0:
            return float("inf")
        return (child.total_value / child.visit_counts) + self.C * np.sqrt(np.log(parent.visit_counts) / child.visit_counts)


    # Choose a snowcap leaf node
    def select(self):
        node = self.root

        while len(node.address) < self.max_depth:

            if len(node.children) != 2:
                return node  # stop at unexplored child

            # choose the child with max UCB
            right_score = self.UCB_score(node.children["R"], node)
            left_score = self.UCB_score(node.children["L"], node)

            if right_score >= left_score:
              node = node.children["R"]
            else:
              node=  node.children["L"]

        return node

    # Choose a random unexplored child from snowcap leaf node to start roll out at
    def expand(self, node):
        if len(node.address) == self.max_depth:
            return node

        unexplored = [a for a in ["L", "R"] if a not in node.children]

        if len(unexplored) == 1:
          choice = unexplored[0]
        else:
          choice = self.rng.choice(unexplored)

        child = Node(node.address + choice, parent=node)
        node.children[choice] = child

        return child

    # Perform a random roll out from child of snowcap leaf node
    def random_roll_out(self, child):
        address = child.address
        while len(address) < self.max_depth:
            choice = self.rng.choice(["L", "R"])
            address += choice
        return self.leaf_values[address]

    # Backpropagate reward value back up from start of roll out to root
    def back_prop(self, child, reward):
        node = child
        while node is not None:
            node.total_value += reward
            node.visit_counts += 1
            self.visited_nodes.add(node)
            node = node.parent

    # Run MCTS algorithm on a particular root till budget exhausted
    def MCTS_on_a_root(self):
        for _ in range(self.max_iterations_per_root):
            leaf = self.select()
            child = self.expand(leaf)
            reward = 0
            for _ in range(self.num_rollouts):
              reward += self.random_roll_out(child)
            reward = reward / self.num_rollouts
            self.back_prop(child, reward)


# Run the MCTS algorithm
def run_MCTS(max_iterations_per_root, C, leaf_values, max_depth, num_rollouts):
    mcts = MCTS(max_depth, C, leaf_values, max_iterations_per_root, num_rollouts)

    for _ in range(max_depth):
        mcts.MCTS_on_a_root()

        # choose new root = best child
        left = mcts.root.children.get("L")
        right = mcts.root.children.get("R")

        if mcts.UCB_score(right, mcts.root) >= mcts.UCB_score(left, mcts.root):
          mcts.root = right
        else:
          mcts.root = left

        mcts.root.parent = None

    return leaf_values[mcts.root.address], mcts.root.address, mcts.visited_nodes


def edit_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def get_addresses(max_depth):
    addresses = [""]
    for _ in range(max_depth):
        new = []
        for addr in addresses:
            new.append(addr + "L")
            new.append(addr + "R")
        addresses = new
    return addresses

# Get the scores of all the leaves
def get_leaf_scores(target_address, max_depth):
    all_addresses = get_addresses(max_depth)
    ed = {addr: edit_distance(addr, target_address) for addr in all_addresses}

    B = 10
    tau = max(ed.values()) / 5

    leaf_scores = {}
    for addr in all_addresses:
        leaf_scores[addr] = B * np.exp(-(ed[addr]/tau)) + np.random.normal()

    return leaf_scores
