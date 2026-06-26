import numpy as np
import matplotlib.pyplot as plt
from MCTS_implementation import Node, MCTS, run_MCTS, edit_distance, get_addresses, get_leaf_scores

# Compute the 4 statistics for a run
def compute_metrics(final_value, final_leaf, leaf_values, visited_nodes):
    all_leaves = list(leaf_values.values())
    optimal = max(all_leaves)
    top10_cutoff = np.percentile(all_leaves, 99)

    return {
        "efficiency": (len(visited_nodes) / (2**(max_depth+1) - 1))*100,
        "effectiveness": final_value / optimal,
        "reliability": 1.0 if leaf_values[final_leaf] >= top10_cutoff else 0.0,
        "exploration_vs_exploitation": np.mean([node.visit_counts for node in visited_nodes])
    }

# Run the experiments with different C values
def run_experiments():
    global max_depth
    max_depth = 20
    num_rollouts = 5
    iters_per_root = 50
    N_trials_per_param = 1000

    target = "LLLLLLLLLLRRRRRRRRRR" # Random target node
    if len(target) != 20:
      print(f"ERROR NOT RIGHT LENGTH TARGET")
    leaf_values = get_leaf_scores(target, max_depth)

    C_values = [0.1, 0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, 5, 7.5, 10]



    def sweep(param_list ):
        results = {p: [] for p in param_list}

        for p in param_list:
          print(f"running C value {p}")
          for i in range(N_trials_per_param):
            C = p
            final_value, leaf, visited = run_MCTS(iters_per_root, C, leaf_values, max_depth, num_rollouts)
            metrics = compute_metrics(final_value, leaf, leaf_values, visited)
            results[p].append(metrics)

        # Average results over trials
        summary = {}
        for p in param_list:
            avg = {
                key: np.mean([r[key] for r in results[p]])
                for key in results[p][0]
            }
            summary[p] = avg
        return summary

    sweepC = sweep(C_values)

    # Make plots
    def plot_results(summary, title_prefix):
        params = list(summary.keys())
        eff = [summary[p]["efficiency"] for p in params]
        effec = [summary[p]["effectiveness"] for p in params]
        rel = [summary[p]["reliability"] for p in params]
        explo = [summary[p]["exploration_vs_exploitation"] for p in params]

        plt.figure(figsize=(12, 8))
        plt.plot(params, eff, marker='o')
        plt.title(f"{title_prefix}: Efficiency")
        plt.xlabel("C value")
        plt.ylabel("percentage of tree nodes visited")
        plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(params, effec, marker='o')
        plt.title(f"{title_prefix}: Effectiveness")
        plt.xlabel("C value")
        plt.ylabel("final value / optimal value")
        plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(params, rel, marker='o')
        plt.title(f"{title_prefix}: Reliability")
        plt.xlabel("C value")
        plt.ylabel("fraction of trials final score in top 1%")
        plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(params, explo, marker='o')
        plt.title(f"{title_prefix}: Exploration vs Exploitation")
        plt.xlabel("C value")
        plt.ylabel("average # of times a visited node was visited")
        plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])
        plt.show()

    plot_results(sweepC, "Effect of C")

run_experiments()
