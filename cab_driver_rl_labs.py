"""
UNIT V: RL-Based Cab Driver Recommendation System
=================================================

Five beginner-friendly laboratory implementations in one Python file.

Requirements:
    pip install pandas numpy matplotlib

Run:
    python cab_driver_rl_labs.py

The menu lets you run each lab independently:
    1. Cab aggregator environment simulation
    2. Rule-based driver recommendation
    3. Multi-Armed Bandit (epsilon-greedy)
    4. Q-Learning driver recommendation
    5. Driver retention mini-project

All data and rewards are simulated for teaching purposes. This is not a
production cab-booking or driver-allocation system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared simulated driver data
# ---------------------------------------------------------------------------

DRIVERS = pd.DataFrame(
    {
        "Driver": ["D1", "D2", "D3", "D4", "D5"],
        "Rating": [4.8, 4.2, 4.5, 4.9, 4.0],
        "Acceptance_Rate": [0.90, 0.70, 0.80, 0.95, 0.60],
        "Waiting_Time": [5, 8, 6, 4, 10],  # minutes
        "Available": [1, 1, 1, 1, 1],
    }
)

# Hidden probabilities used only by the simulator in Lab 3.
TRUE_ACCEPTANCE_PROBABILITIES = np.array([0.90, 0.70, 0.50, 0.80, 0.60])


def heading(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# LAB 1: Simulating a cab aggregator environment
# ---------------------------------------------------------------------------

def lab1_environment():
    heading("LAB 1: CAB AGGREGATOR ENVIRONMENT SIMULATION")

    print("\nDriver information:")
    print(DRIVERS.to_string(index=False))

    locations = ["Location A", "Location B", "Location C", "Location D"]
    pickup = np.random.choice(locations)
    drop = np.random.choice([loc for loc in locations if loc != pickup])

    print("\nNew ride request generated")
    print(f"Pickup location : {pickup}")
    print(f"Drop location   : {drop}")

    available = DRIVERS[DRIVERS["Available"] == 1]
    print("\nAvailable drivers:")
    print(available[["Driver", "Rating", "Acceptance_Rate", "Waiting_Time"]]
          .to_string(index=False))

    print("\nTask: Change driver availability and generate another ride request.")


# ---------------------------------------------------------------------------
# LAB 2: Rule-based recommendation (baseline for comparison)
# ---------------------------------------------------------------------------

def lab2_rule_based():
    heading("LAB 2: RULE-BASED DRIVER RECOMMENDATION")

    df = DRIVERS[DRIVERS["Available"] == 1].copy()

    # Higher rating and acceptance rate are better; lower waiting time is better.
    df["Score"] = (
        (df["Rating"] / 5.0) * 0.40
        + df["Acceptance_Rate"] * 0.40
        + (1.0 - df["Waiting_Time"] / 15.0) * 0.20
    )

    df = df.sort_values("Score", ascending=False)
    print("\nDriver recommendation scores:")
    print(df[["Driver", "Rating", "Acceptance_Rate",
              "Waiting_Time", "Score"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\nRecommended driver: {best['Driver']}")
    print(f"Recommendation score: {best['Score']:.3f}")


# ---------------------------------------------------------------------------
# LAB 3: Multi-Armed Bandit using epsilon-greedy
# ---------------------------------------------------------------------------

def lab3_multi_armed_bandit(episodes=500, epsilon=0.10, seed=42):
    heading("LAB 3: MULTI-ARMED BANDIT - EPSILON-GREEDY")

    rng = np.random.default_rng(seed)
    n_drivers = len(DRIVERS)
    counts = np.zeros(n_drivers, dtype=int)
    total_rewards = np.zeros(n_drivers, dtype=float)
    reward_history = []

    for ride in range(episodes):
        # Explore with probability epsilon; otherwise exploit current best estimate.
        if rng.random() < epsilon:
            chosen = int(rng.integers(n_drivers))
        else:
            estimates = np.divide(
                total_rewards,
                counts,
                out=np.zeros(n_drivers, dtype=float),
                where=counts > 0,
            )
            # Prefer an untried driver until each driver has been tried.
            untried = np.where(counts == 0)[0]
            chosen = int(rng.choice(untried)) if len(untried) else int(np.argmax(estimates))

        accepted = rng.random() < TRUE_ACCEPTANCE_PROBABILITIES[chosen]
        reward = 1 if accepted else 0

        counts[chosen] += 1
        total_rewards[chosen] += reward
        reward_history.append(reward)

    results = pd.DataFrame(
        {
            "Driver": DRIVERS["Driver"],
            "Requests_Assigned": counts,
            "Accepted_Rides": total_rewards.astype(int),
            "Estimated_Acceptance": np.divide(
                total_rewards, counts,
                out=np.zeros(n_drivers), where=counts > 0
            ),
        }
    )

    print("\nBandit results:")
    print(results.to_string(index=False))

    best_index = int(np.argmax(results["Estimated_Acceptance"].to_numpy()))
    print(f"\nBest estimated driver: {DRIVERS.iloc[best_index]['Driver']}")

    window = min(25, episodes)
    rolling = pd.Series(reward_history).rolling(window).mean()
    plt.figure(figsize=(9, 4))
    plt.plot(rolling)
    plt.title(f"Lab 3: Rolling acceptance reward (window={window})")
    plt.xlabel("Ride request")
    plt.ylabel("Average reward")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# LAB 4: Q-Learning across demand states and driver actions
# ---------------------------------------------------------------------------

def lab4_q_learning(episodes=3000, seed=42):
    heading("LAB 4: Q-LEARNING FOR DRIVER RECOMMENDATIONS")

    rng = np.random.default_rng(seed)

    # States: 0 = low demand, 1 = medium demand, 2 = high demand.
    state_names = ["Low demand", "Medium demand", "High demand"]
    n_states = 3
    n_actions = len(DRIVERS)

    # Simulated acceptance probabilities vary with demand.
    acceptance_by_state = np.array(
        [
            [0.90, 0.70, 0.60, 0.85, 0.65],
            [0.85, 0.72, 0.75, 0.90, 0.62],
            [0.75, 0.65, 0.80, 0.85, 0.70],
        ]
    )

    Q = np.zeros((n_states, n_actions))
    alpha = 0.10
    gamma = 0.90
    epsilon = 0.15

    for _ in range(episodes):
        state = int(rng.integers(n_states))

        if rng.random() < epsilon:
            action = int(rng.integers(n_actions))
        else:
            action = int(np.argmax(Q[state]))

        accepted = rng.random() < acceptance_by_state[state, action]
        reward = 10 if accepted else -5

        # Illustrative customer experience adjustment based on driver rating.
        rating = DRIVERS.iloc[action]["Rating"]
        if accepted and rating >= 4.7:
            reward += 2

        next_state = int(rng.integers(n_states))
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

    print("\nLearned Q-table (rows = demand states, columns = drivers):")
    print(pd.DataFrame(Q, index=state_names, columns=DRIVERS["Driver"]).round(2))

    print("\nRecommended driver by demand:")
    for state in range(n_states):
        best_action = int(np.argmax(Q[state]))
        print(f"{state_names[state]}: {DRIVERS.iloc[best_action]['Driver']}")

    print("\nNote: Rewards and transitions are simulated for learning purposes.")


# ---------------------------------------------------------------------------
# LAB 5: Driver retention mini-project
# ---------------------------------------------------------------------------

def run_retention_simulation(use_rl=True, episodes=1000, seed=7):
    """Run a simple simulated ride-allocation experiment.

    The RL policy learns a preferred driver for a low/medium/high demand state.
    A small fairness bonus encourages the agent to avoid repeatedly assigning
    every ride to the same driver. Retention is represented by a simulated
    satisfaction score, not a real-world prediction.
    """
    rng = np.random.default_rng(seed)
    n_drivers = len(DRIVERS)
    n_states = 3

    acceptance = np.array(
        [
            [0.90, 0.70, 0.60, 0.85, 0.65],
            [0.85, 0.72, 0.75, 0.90, 0.62],
            [0.75, 0.65, 0.80, 0.85, 0.70],
        ]
    )

    assignments = np.zeros(n_drivers, dtype=int)
    accepted_counts = np.zeros(n_drivers, dtype=int)
    satisfaction_points = np.zeros(n_drivers, dtype=float)
    Q = np.zeros((n_states, n_drivers))

    alpha, gamma, epsilon = 0.10, 0.90, 0.15

    for _ in range(episodes):
        state = int(rng.integers(n_states))

        if use_rl and rng.random() < epsilon:
            action = int(rng.integers(n_drivers))
        elif use_rl:
            action = int(np.argmax(Q[state]))
        else:
            # Baseline: always select the driver with the highest fixed score.
            action = int(np.argmax(
                DRIVERS["Rating"].to_numpy() * 0.4
                + DRIVERS["Acceptance_Rate"].to_numpy() * 0.6
            ))

        assignments[action] += 1
        accepted = rng.random() < acceptance[state, action]

        if accepted:
            accepted_counts[action] += 1
            satisfaction_points[action] += 1.0
            reward = 10
        else:
            reward = -5

        # Fairness bonus: small positive reward when a less-used driver is selected.
        average_assignments = assignments.sum() / n_drivers
        if assignments[action] <= average_assignments:
            reward += 2

        # Overload penalty when one driver receives substantially more than average.
        if assignments[action] > average_assignments * 1.5:
            reward -= 3

        next_state = int(rng.integers(n_states))
        if use_rl:
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

    satisfaction = np.divide(
        satisfaction_points,
        assignments,
        out=np.zeros(n_drivers, dtype=float),
        where=assignments > 0,
    ) * 100

    return pd.DataFrame(
        {
            "Driver": DRIVERS["Driver"],
            "Assigned_Rides": assignments,
            "Accepted_Rides": accepted_counts,
            "Acceptance_Percent": np.divide(
                accepted_counts, assignments,
                out=np.zeros(n_drivers, dtype=float),
                where=assignments > 0,
            ) * 100,
            "Simulated_Satisfaction_Percent": satisfaction,
        }
    )


def lab5_driver_retention():
    heading("LAB 5: DRIVER RETENTION MINI-PROJECT")

    print("\nRunning two simulated strategies for comparison...")
    baseline = run_retention_simulation(use_rl=False)
    rl_results = run_retention_simulation(use_rl=True)

    print("\nA. Fixed rule-based allocation:")
    print(baseline.round(1).to_string(index=False))

    print("\nB. RL-based allocation:")
    print(rl_results.round(1).to_string(index=False))

    print("\nTotal rides assigned:")
    print("Rule-based:", int(baseline["Assigned_Rides"].sum()))
    print("RL-based  :", int(rl_results["Assigned_Rides"].sum()))

    print("\nRide distribution comparison:")
    comparison = pd.DataFrame(
        {
            "Driver": DRIVERS["Driver"],
            "Rule_Based": baseline["Assigned_Rides"],
            "RL_Based": rl_results["Assigned_Rides"],
        }
    )
    print(comparison.to_string(index=False))

    comparison.plot(
        x="Driver", y=["Rule_Based", "RL_Based"],
        kind="bar", figsize=(9, 4), rot=0
    )
    plt.title("Lab 5: Ride allocation comparison")
    plt.xlabel("Driver")
    plt.ylabel("Number of assigned rides")
    plt.tight_layout()
    plt.show()

    print(
        "\nDiscussion: Compare how evenly rides are distributed and how many "
        "rides are accepted. This toy simulation illustrates retention-related "
        "signals; it does not prove actual long-term driver retention."
    )


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    labs = {
        "1": lab1_environment,
        "2": lab2_rule_based,
        "3": lab3_multi_armed_bandit,
        "4": lab4_q_learning,
        "5": lab5_driver_retention,
    }

    while True:
        heading("UNIT V: RL-BASED CAB DRIVER RECOMMENDATION - LAB MENU")
        print("1. Lab 1 - Simulate cab aggregator environment")
        print("2. Lab 2 - Rule-based driver recommendation")
        print("3. Lab 3 - Multi-Armed Bandit (epsilon-greedy)")
        print("4. Lab 4 - Q-Learning driver recommendation")
        print("5. Lab 5 - Driver retention mini-project")
        print("0. Exit")

        choice = input("\nEnter lab number (0-5): ").strip()

        if choice == "0":
            print("Exiting the lab program. All the best!")
            break
        elif choice in labs:
            labs[choice]()
        else:
            print("Invalid choice. Please enter a number from 0 to 5.")

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
