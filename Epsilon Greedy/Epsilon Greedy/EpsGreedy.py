from packages import *

# Epsilon Greedy
"""
Implementing Epsilon Greedy algorithm in the standard as well as annealing ways
"""

"""Implementing standard epsilon-Greedy algorithm.

    """


class EpsilonGreedy:

    def __init__(self, eps, count=None, value=None):
        self.eps = eps #the probability of explore the available arms
        self.count = count #the quantity of pulls for every arms
        self.value = value #the average quantity of reward we obtain from every arms

        """making value and count arrays with zeros"""

    def initialize(self, n_arms):
        self.count = [0 for col in range(n_arms)]
        self.value = [0.0 for col in range(n_arms)]

    # Picking  best arm. argmax is returning index of max value

    def select_arm(self):
        max_index = np.random.random()
        if max_index < self.eps:
            return np.random.randint(0, len(self.value))
        return np.argmax(self.value)

        """For chosen arm updating number and est. value of rewards."""
    def update(self, chosen_arm, reward):
        self.count[chosen_arm] += 1
        n = self.count[chosen_arm]

        # Recompute the estimated value of chosen arm using new reward
        value = self.value[chosen_arm]
        new_value = ((n - 1) / n) * value + reward * (1/n)
        self.value[chosen_arm] = new_value

    def __ne__(self) -> str:
        return "eps = 0.3"
