from collections import namedtuple
import random
import numpy as np
from collections import namedtuple


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.position = 0
        self.length = 0

    def add(self, priority, data):
        idx = self.position + self.capacity - 1
        self.data[self.position] = data
        self.update(idx, priority)
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def total_priority(self):
        return self.tree[0]


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.trans_memory = []
        self.position = 0
        self.beta_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.

    def push(self, *args):
        # Give max priority to new transition so it's sampled next
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        data = Transition(*args)
        self.tree.add(max_priority, data)

        if len(self.trans_memory) < self.capacity:
            self.trans_memory.append(data)
        else:
            self.trans_memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get(value)
            priorities.append(priority)
            transitions.append(data)

        # Calculate max weight
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        weights /= weights.max()

        return transitions  # same interface as requested

    def __len__(self):
        return len(self.trans_memory)

    def update_priority(self, idx, error):
        priority = self.get_priority(error)
        self.tree.update(idx, priority)

    def get_priority(self, error):
        return (error + 0.01) ** self.alpha

