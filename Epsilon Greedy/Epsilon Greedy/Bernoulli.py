from packages import *


class BernoulliArm:
    def __init__(self, p):
        self.p = p
        ' p stands for the prob. of obtaining reward for particular area'
        'draw stands for rewards drawn applying Bernoulli prob. dist.'

    def draw(self):
        z = np.random.random()
        if z > self.p:
            return 0.0
        return 1.0
        # An arm rewards you with a value of 1 some percentage of the time
        # and rewards you with a value of 0 the rest of the time.
