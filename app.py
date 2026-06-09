import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from simulation import run_simulation

st.set_page_config(
    page_title="Bandit Recommender",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Multi-Armed Bandit: Real-Time Recommendation Learning")
st.markdown("""
> **Scenario:** You run a news site with 10 articles. Each article has a hidden
> true click rate. Your algorithm must *discover* the best articles by watching
> user behaviour — without knowing the true rates in advance.
""")

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_arms   = st.slider("Number of Articles (Arms)", 5, 20, 10)
    n_rounds = st.slider("Rounds (Recommendations)", 1000, 50000, 10000, step=1000)
    epsilon  = st.slider("Epsilon (ε-Greedy explore rate)", 0.01, 0.5, 0.1)
    alpha    = st.slider("Alpha (LinUCB confidence)", 0.1, 2.0, 0.5)
    run_btn  = st.button("▶ Run Simulation", type="primary", use_container_width=True)

@st.cache_data
def cached_simulation(n_arms, n_rounds, epsilon, alpha):
    return run_simulation(n_arms=n_arms, n_rounds=n_rounds, epsilon=epsilon, alpha=alpha)

if run_btn:
    with st.spinner("Simulating... ⏳"):
        results = cached_simulation(n_arms, n_rounds, epsilon, alpha)
    st.success("Done! Scroll down to see results.")

    rewards    = results["rewards"]
    regrets    = results["regrets"]
    arm_counts = results["arm_counts"]
    true_probs = results["true_probs"]
    best_arm   = results["best_arm"]
    ts_agent   = results["ts_agent"]

    colors = {
        "Epsilon-Greedy": "#e74c3c",
        "UCB1":           "#3498db",
        "Thompson":       "#2ecc71",
        "LinUCB":         "#9b59b6",
    }
    rounds_x = range(1, n_rounds + 1)

    st.header("📉 Cumulative Regret (Lower = Smarter)")
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in colors.items():
        ax.plot(rounds_x, regrets[name], label=name, color=color, linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.info("""
    **Why Thompson Sampling wins on regret:** It represents uncertainty as a full
    Beta probability distribution per arm. Early on, all distributions are wide
    so it explores freely. As data accumulates, distributions narrow around true
    values and exploitation becomes natural. No hyperparameter needed.
    """)

    st.header("📈 Cumulative Reward (Higher = Better)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for name, color in colors.items():
        ax2.plot(rounds_x, rewards[name], label=name, color=color, linewidth=2)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Reward")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    st.header("📊 Final Results Summary")
    summary = []
    for name in colors:
        final_reward = rewards[name][-1]
        final_regret = regrets[name][-1]
        top_arm = np.argmax(arm_counts[name])
        found_best = "✅ Yes" if top_arm == best_arm else "❌ No"
        summary.append({
            "Algorithm":       name,
            "Final Reward":    f"{final_reward:.1f}",
            "Final Regret":    f"{final_regret:.1f}",
            "Best Arm Found?": found_best,
        })
    st.table(summary)

    st.header("🎯 Which Articles Did Each Algorithm Recommend?")
    fig3, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for i, (name, color) in enumerate(colors.items()):
        axes[i].bar(range(n_arms), arm_counts[name], color=color, alpha=0.8)
        axes[i].axvline(best_arm, color="black", linestyle="--",
                        linewidth=2, label=f"Best Arm ({best_arm})")
        axes[i].set_title(name)
        axes[i].set_xlabel("Article #")
        axes[i].set_ylabel("Times Recommended")
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.header("🔬 Thompson Sampling: What It Learned")
    x = np.linspace(0, 1, 300)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    for arm in range(n_arms):
        a = ts_agent.successes[arm]
        b = ts_agent.failures[arm]
        ax4.plot(x, beta_dist.pdf(x, a, b), alpha=0.7,
                 label=f"Arm {arm} (true={true_probs[arm]:.2f})")
    ax4.set_title("Beta Distributions After Full Run — Narrow = Confident")
    ax4.set_xlabel("Estimated Click Probability")
    ax4.set_ylabel("Probability Density")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(alpha=0.3)
    st.pyplot(fig4)
    plt.close()

    st.header("🧑‍💻 Try LinUCB Live — Enter Your User Profile")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: age_group        = st.slider("Age Group",        0.0, 1.0, 0.5)
    with col2: morning          = st.slider("Morning User",     0.0, 1.0, 0.5)
    with col3: tech_reader      = st.slider("Tech Reader",      0.0, 1.0, 0.5)
    with col4: mobile_user      = st.slider("Mobile User",      0.0, 1.0, 0.5)
    with col5: frequent_visitor = st.slider("Frequent Visitor", 0.0, 1.0, 0.5)

    context = np.array([age_group, morning, tech_reader, mobile_user, frequent_visitor])

    from agents import LinUCB as LinUCBAgent
    from environment import ContextualBanditEnvironment
    ctx_env2 = ContextualBanditEnvironment(n_arms=n_arms)
    linucb2  = LinUCBAgent(n_arms=n_arms, alpha=alpha)
    for _ in range(2000):
        ctx = ctx_env2.get_context()
        arm = linucb2.select_arm(ctx)
        reward, _ = ctx_env2.pull(arm, ctx)
        linucb2.update(arm, ctx, reward)

    recommended = linucb2.select_arm(context)
    A_inv = np.linalg.inv(linucb2.A[recommended])
    theta = A_inv @ linucb2.b[recommended]
    confidence = float(alpha * np.sqrt(context @ A_inv @ context))

    st.success(f"📰 LinUCB recommends: **Article #{recommended}**")
    st.metric("Predicted Click Probability",
              f"{max(0, min(1, float(theta @ context))):.2%}")
    st.metric("Confidence Bonus", f"{confidence:.4f}")

else:
    st.info("👈 Adjust settings in the sidebar and click **▶ Run Simulation** to start.")
    st.markdown("""
    ### What Each Algorithm Does
    | Algorithm | Core Idea | Best For |
    |-----------|-----------|----------|
    | **ε-Greedy** | Random explore 10% of the time | Simple baseline |
    | **UCB1** | Explore arms with high uncertainty | Math guarantees |
    | **Thompson Sampling** | Bayesian — sample from belief distributions | Best regret |
    | **LinUCB** | Personalise using user features | Production systems |
    """)