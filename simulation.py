import numpy as np
import matplotlib.pyplot as plt
from environment import BanditEnvironment, ContextualBanditEnvironment
from agents import EpsilonGreedy, UCB1, ThompsonSampling, LinUCB

def run_simulation(n_arms=10, n_rounds=10000, epsilon=0.1, alpha=0.5, seed=42):
    env = BanditEnvironment(n_arms=n_arms, seed=seed)
    ctx_env = ContextualBanditEnvironment(n_arms=n_arms, seed=seed)

    agents = {
        "Epsilon-Greedy": EpsilonGreedy(n_arms, epsilon=epsilon),
        "UCB1":           UCB1(n_arms),
        "Thompson":       ThompsonSampling(n_arms),
    }
    linucb = LinUCB(n_arms, alpha=alpha)

    rewards    = {name: [] for name in agents}
    rewards["LinUCB"] = []
    regrets    = {name: [] for name in agents}
    regrets["LinUCB"] = []
    arm_counts = {name: np.zeros(n_arms) for name in list(agents.keys()) + ["LinUCB"]}

    cum_rewards = {name: 0 for name in list(agents.keys()) + ["LinUCB"]}
    cum_regrets = {name: 0 for name in list(agents.keys()) + ["LinUCB"]}

    for t in range(1, n_rounds + 1):
        context = ctx_env.get_context()

        for name, agent in agents.items():
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            cum_rewards[name] += reward
            cum_regrets[name] += (env.best_prob - env.true_probs[arm])
            arm_counts[name][arm] += 1
            rewards[name].append(cum_rewards[name])
            regrets[name].append(cum_regrets[name])

        arm = linucb.select_arm(context)
        reward, _ = ctx_env.pull(arm, context)
        linucb.update(arm, context, reward)
        cum_rewards["LinUCB"] += reward
        cum_regrets["LinUCB"] += float(
            np.max(ctx_env.arm_weights @ context) - ctx_env.arm_weights[arm] @ context
        )
        arm_counts["LinUCB"][arm] += 1
        rewards["LinUCB"].append(cum_rewards["LinUCB"])
        regrets["LinUCB"].append(cum_regrets["LinUCB"])

    return {
        "rewards":    rewards,
        "regrets":    regrets,
        "arm_counts": arm_counts,
        "true_probs": env.true_probs,
        "best_arm":   env.best_arm,
        "ts_agent":   agents["Thompson"],
        "n_rounds":   n_rounds,
        "n_arms":     n_arms,
    }


def plot_results(results):
    rewards    = results["rewards"]
    regrets    = results["regrets"]
    arm_counts = results["arm_counts"]
    n_rounds   = results["n_rounds"]

    colors = {
        "Epsilon-Greedy": "#e74c3c",
        "UCB1":           "#3498db",
        "Thompson":       "#2ecc71",
        "LinUCB":         "#9b59b6",
    }
    rounds = range(1, n_rounds + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in colors.items():
        ax.plot(rounds, regrets[name], label=name, color=color, linewidth=2)
    ax.set_title("Cumulative Regret Over Time (lower = smarter)", fontsize=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("regret_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in colors.items():
        ax.plot(rounds, rewards[name], label=name, color=color, linewidth=2)
    ax.set_title("Cumulative Reward Over Time (higher = better)", fontsize=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_curves.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (name, color) in enumerate(colors.items()):
        axes[i].bar(range(results["n_arms"]), arm_counts[name], color=color, alpha=0.8)
        axes[i].axvline(results["best_arm"], color="black", linestyle="--", label="Best Arm")
        axes[i].set_title(f"{name} — Arm Selection")
        axes[i].set_xlabel("Arm")
        axes[i].set_ylabel("Times Selected")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig("arm_counts.png", dpi=150)
    plt.close()

    from scipy.stats import beta as beta_dist
    x = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(12, 5))
    ts = results["ts_agent"]
    for arm in range(results["n_arms"]):
        a = ts.successes[arm]
        b = ts.failures[arm]
        ax.plot(x, beta_dist.pdf(x, a, b), alpha=0.6,
                label=f"Arm {arm} (true={results['true_probs'][arm]:.2f})")
    ax.set_title("Thompson Sampling: Beta Distributions After Full Run", fontsize=14)
    ax.set_xlabel("Estimated Click Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("beta_distributions.png", dpi=150)
    plt.close()
    print("✅ All 4 charts saved!")


if __name__ == "__main__":
    print("Running simulation... (~10 seconds)")
    results = run_simulation()
    plot_results(results)
    print(f"Best arm: {results['best_arm']} | True prob: {results['true_probs'][results['best_arm']]:.3f}")