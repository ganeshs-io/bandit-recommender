<div align="center">

# 🎰 Multi-Armed Bandit Recommender

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy)](https://numpy.org)

**Benchmarking 4 bandit algorithms on a simulated news recommendation environment**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-FF4B4B?style=for-the-badge)](https://bandit-recommender.streamlit.app)

</div>

---

## 📌 Overview

Amazon's product recommendation and ad-ranking systems run on bandit algorithms in production. This project implements 4 strategies from scratch and races them over 10,000 recommendation rounds on a simulated news platform with 10 articles — each having a hidden true click rate the algorithm must discover.

---

## 🏆 Results

| Algorithm | Final Reward | Final Regret | Best Arm Found |
|:----------|:-----------:|:------------:|:--------------:|
| ε-Greedy | 6,199 | 6,441 | ✅ |
| UCB1 | 8,119 | 523 | ✅ |
| Thompson Sampling | **8,581** | **82** | ✅ |
| LinUCB (Contextual) | 7,187 | 4,902 | — |

> **Thompson Sampling achieves 98% lower regret than ε-Greedy** by representing uncertainty as a full Beta probability distribution per arm rather than a single point estimate.

---

## 🧠 The 4 Algorithms

**ε-Greedy**
Explores randomly 10% of the time, exploits the best known arm 90% of the time. Simple but wastes a fixed percentage of rounds forever.

**UCB1 (Upper Confidence Bound)**
Picks the arm maximising: `average reward + √(2 × ln(total) / count)`. Automatically balances exploration and exploitation with no randomness. Mathematically principled.

**Thompson Sampling**
Maintains a Beta(successes+1, failures+1) distribution per arm. Each round, samples from every distribution and picks the highest. Bayesian and parameter-free. Achieves lowest regret in practice.

**LinUCB (Contextual Bandit)**
The industry-grade algorithm. Each user has a 5-dimensional context vector. LinUCB fits a ridge regression model per arm to personalise recommendations, then adds an uncertainty bonus like UCB.

---

## 📊 Visualizations

### Cumulative Regret — Lower is Smarter
*Thompson Sampling (green) stays flat. ε-Greedy (red) grows linearly forever.*

### Arm Selection Frequency
*Thompson Sampling concentrates almost all recommendations on the single best arm after convergence.*

### Beta Distributions — What Thompson Sampling Learned
*Narrow tall peaks = confident. Wide flat curves = uncertain. Peaks align with true click rates.*

---

## 🎮 Live Demo Features

- Adjust number of articles (5–20), rounds (1k–50k), epsilon, and alpha
- Watch all 4 algorithms race in real time
- Try LinUCB live — enter your user profile via sliders and get a personalised recommendation

---

## 🛠️ Tech Stack

`Python` · `NumPy` · `Matplotlib` · `SciPy` · `Streamlit`

---

## 🚀 Run Locally

    git clone https://github.com/ganeshs-io/bandit-recommender.git
    cd bandit-recommender
    pip install -r requirements.txt
    python -m streamlit run app.py
