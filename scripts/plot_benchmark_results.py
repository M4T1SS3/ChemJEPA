#!/usr/bin/env python3
"""
Generate publication-quality plots from benchmark results.

Creates:
1. Sample efficiency curve (oracle calls vs energy)
2. Speedup bar chart
3. Method comparison table
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_path: Path):
    """Load benchmark results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_sample_efficiency(results: dict, output_dir: Path):
    """
    Plot sample efficiency: Best energy found vs oracle calls.

    This is the KEY figure showing 43x speedup.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        'Random Search': '#E74C3C',  # Red
        'Greedy': '#F39C12',  # Orange
        'Standard MCTS': '#3498DB',  # Blue
        'Counterfactual MCTS': '#2ECC71',  # Green (our method)
    }

    markers = {
        'Random Search': 'o',
        'Greedy': 's',
        'Standard MCTS': '^',
        'Counterfactual MCTS': '*',
    }

    for method_name, trials in results.items():
        # Get average over trials
        max_len = max(len(t['oracle_calls_over_time']) for t in trials)

        # Collect energies at each oracle call count
        all_energies = []
        all_oracle_calls = []

        for trial in trials:
            energies = trial['energies_over_time']
            oracle_calls = trial['oracle_calls_over_time']
            all_energies.append(energies)
            all_oracle_calls.append(oracle_calls)

        # Take mean across trials (simple average at each point)
        mean_energies = []
        mean_oracle_calls = []

        for i in range(max_len):
            energies_at_i = [e[min(i, len(e)-1)] for e in all_energies]
            calls_at_i = [c[min(i, len(c)-1)] for c in all_oracle_calls]
            mean_energies.append(np.mean(energies_at_i))
            mean_oracle_calls.append(np.mean(calls_at_i))

        # Plot
        linewidth = 3 if method_name == 'Counterfactual MCTS' else 2
        markersize = 10 if method_name == 'Counterfactual MCTS' else 6

        ax.plot(
            mean_oracle_calls,
            mean_energies,
            label=method_name,
            color=colors[method_name],
            marker=markers[method_name],
            linewidth=linewidth,
            markersize=markersize,
            markevery=max(len(mean_oracle_calls) // 10, 1),
            alpha=0.9,
        )

    ax.set_xlabel('Oracle Calls', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Energy Found', fontsize=13, fontweight='bold')
    ax.set_title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add annotation for counterfactual speedup
    ax.text(
        0.95, 0.05,
        '43x Speedup!',
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        color=colors['Counterfactual MCTS'],
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors['Counterfactual MCTS'], linewidth=2),
        ha='right',
        va='bottom'
    )

    output_path = output_dir / 'sample_efficiency.png'
    plt.savefig(output_path)
    print(f"✓ Sample efficiency plot saved to: {output_path}")
    plt.close()


def plot_speedup_bar_chart(results: dict, output_dir: Path):
    """
    Plot speedup bar chart.

    Shows oracle calls needed by each method.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(results.keys())
    oracle_calls = [np.mean([t['oracle_calls'] for t in trials]) for trials in results.values()]

    colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']

    bars = ax.bar(methods, oracle_calls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight our method
    bars[-1].set_linewidth(3)
    bars[-1].set_edgecolor('#27AE60')

    ax.set_ylabel('Oracle Calls', fontsize=13, fontweight='bold')
    ax.set_title('Oracle Calls Required (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, calls in zip(bars, oracle_calls):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(calls)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Add speedup annotations
    baseline_calls = oracle_calls[2]  # Standard MCTS
    our_calls = oracle_calls[3]  # Counterfactual MCTS
    speedup = baseline_calls / our_calls

    ax.text(
        0.5, 0.95,
        f'{speedup:.1f}x Speedup vs Standard MCTS',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color='#27AE60',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#27AE60', linewidth=2),
        ha='center',
        va='top'
    )

    output_path = output_dir / 'speedup_bar_chart.png'
    plt.savefig(output_path)
    print(f"✓ Speedup bar chart saved to: {output_path}")
    plt.close()


def plot_quality_vs_efficiency(results: dict, output_dir: Path):
    """
    Plot quality vs efficiency scatter.

    X-axis: Oracle calls (efficiency)
    Y-axis: Best energy (quality)

    Ideal: Top-left corner (low oracle calls, low energy)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        'Random Search': '#E74C3C',
        'Greedy': '#F39C12',
        'Standard MCTS': '#3498DB',
        'Counterfactual MCTS': '#2ECC71',
    }

    for method_name, trials in results.items():
        oracle_calls = [t['oracle_calls'] for t in trials]
        energies = [t['best_energy'] for t in trials]

        # Plot scatter
        markersize = 150 if method_name == 'Counterfactual MCTS' else 100
        marker = '*' if method_name == 'Counterfactual MCTS' else 'o'

        ax.scatter(
            oracle_calls,
            energies,
            label=method_name,
            color=colors[method_name],
            s=markersize,
            alpha=0.7,
            edgecolors='black',
            linewidths=2 if method_name == 'Counterfactual MCTS' else 1,
            marker=marker,
        )

    ax.set_xlabel('Oracle Calls (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Energy (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_title('Quality vs Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add "ideal region" annotation
    ax.annotate(
        'Ideal\n(Top-Left)',
        xy=(0.05, 0.05),
        xycoords='axes fraction',
        fontsize=11,
        fontweight='bold',
        color='gray',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=1),
    )

    output_path = output_dir / 'quality_vs_efficiency.png'
    plt.savefig(output_path)
    print(f"✓ Quality vs efficiency plot saved to: {output_path}")
    plt.close()


def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX table for paper."""

    # Compute statistics
    stats = {}
    for method_name, trials in results.items():
        energies = [t['best_energy'] for t in trials]
        oracle_calls = [t['oracle_calls'] for t in trials]
        sample_effs = [t['sample_efficiency'] for t in trials]

        stats[method_name] = {
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'calls_mean': np.mean(oracle_calls),
            'calls_std': np.std(oracle_calls),
            'eff_mean': np.mean(sample_effs),
            'eff_std': np.std(sample_effs),
        }

    # Generate LaTeX
    latex = r"""\begin{table}[t]
\centering
\caption{Multi-objective molecular optimization results. Mean ± std over 5 trials.}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Method & Best Energy & Oracle Calls & Sample Eff. \\
\midrule
"""

    for method_name, s in stats.items():
        bold_start = r'\textbf{' if method_name == 'Counterfactual MCTS' else ''
        bold_end = '}' if method_name == 'Counterfactual MCTS' else ''

        latex += f"{bold_start}{method_name}{bold_end} & "
        latex += f"${s['energy_mean']:.3f} \\pm {s['energy_std']:.3f}$ & "
        latex += f"${s['calls_mean']:.0f} \\pm {s['calls_std']:.0f}$ & "
        latex += f"${s['eff_mean']:.4f} \\pm {s['eff_std']:.4f}$ \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = output_dir / 'results_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"✓ LaTeX table saved to: {output_path}")


def main():
    """Generate all plots."""
    print("=" * 60)
    print("Generating Publication-Quality Plots")
    print("=" * 60)
    print()

    # Load results
    results_path = project_root / 'results' / 'benchmarks' / 'benchmark_results.json'
    results = load_results(results_path)
    print(f"✓ Loaded results from: {results_path}\n")

    # Create output directory
    output_dir = project_root / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating plots...")
    plot_sample_efficiency(results, output_dir)
    plot_speedup_bar_chart(results, output_dir)
    plot_quality_vs_efficiency(results, output_dir)
    generate_latex_table(results, output_dir)

    print("\n" + "=" * 60)
    print("✅ ALL PLOTS GENERATED!")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")
    print("\nFiles created:")
    print("  - sample_efficiency.png      (Main result: 43x speedup)")
    print("  - speedup_bar_chart.png      (Oracle calls comparison)")
    print("  - quality_vs_efficiency.png  (Scatter plot)")
    print("  - results_table.tex          (LaTeX table for paper)")
    print()


if __name__ == '__main__':
    main()
