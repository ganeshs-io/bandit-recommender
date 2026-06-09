import numpy as np

class BanditEnvironment:
    def __init__(self, n_arms=10, seed=42):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.true_probs = np.random.uniform(0.1, 0.9, n_arms)
        self.best_arm = np.argmax(self.true_probs)
        self.best_prob = self.true_probs[self.best_arm]

    def pull(self, arm_index):
        return np.random.binomial(1, self.true_probs[arm_index])


class ContextualBanditEnvironment:
    def __init__(self, n_arms=10, n_features=5, seed=42):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.n_features = n_features
        self.arm_weights = np.random.randn(n_arms, n_features) * 0.5

    def get_context(self):
        return np.random.randn(self.n_features)

    def pull(self, arm_index, context):
        score = np.dot(context, self.arm_weights[arm_index])
        prob = 1 / (1 + np.exp(-score))
        return np.random.binomial(1, prob), prob