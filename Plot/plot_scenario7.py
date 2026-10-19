import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_scenario_files(pattern, scenario_label):
    file_list = sorted(glob.glob(pattern))
    frames = []
    for file in file_list:
        df = pd.read_csv(file)
        df = df.rename(columns={
            'workload_variance': 'workload_var',
            'workload_var': 'workload_var'
        })
        df['Scenario'] = scenario_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_combined_data(df_combined):
    agg_df = (
        df_combined
        .groupby(['Scenario', 'placer_name', 'num_edge_servers'])
        .agg(
            avg_distance_mean=('avg_distance', 'mean'),
            avg_distance_std=('avg_distance', 'std'),
            workload_var_mean=('workload_var', 'mean'),
            workload_var_std=('workload_var', 'std'),
            fitness_mean=('fitness', 'mean'),
            fitness_std=('fitness', 'std')
        )
        .reset_index()
    )
    return agg_df


def save_errorbar_plot(summary_df, y_mean, y_std, title, ylabel, filename, use_log=False):
    plt.figure(figsize=(18, 10))
    algorithms = sorted(summary_df['placer_name'].unique())
    base_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728', '#17becf']
    pu_colors = ['#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    palette = {alg: (base_colors[i], pu_colors[i]) for i, alg in enumerate(algorithms)}

    for (placer_name, scenario), group in summary_df.groupby(['placer_name', 'Scenario']):
        group = group.sort_values('num_edge_servers')
        linestyle = '-' if scenario == 'Without PU' else '--'
        color = palette[placer_name][0] if scenario == 'Without PU' else palette[placer_name][1]
        plt.errorbar(
            group['num_edge_servers'],
            group[y_mean],
            yerr=group[y_std],
            fmt='o',
            capsize=3,
            elinewidth=1,
            markersize=5,
            linewidth=1.5,
            alpha=0.9,
            color=color,
            markeredgecolor='black',
            markeredgewidth=0.8,
            linestyle=linestyle,
            label=f'{placer_name} ({scenario})'
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Number of Edge Servers', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    if use_log:
        plt.yscale('log')

    plt.legend(
        title='Algorithm (Scenario)',
        fontsize='11',
        title_fontsize='13',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_averages_to_csv(summary_df, filename='scenario7_8_averages.csv'):
    """Export the average values to CSV in linear scale."""
    export_df = summary_df[['Scenario', 'placer_name', 'num_edge_servers', 'avg_distance_mean', 'workload_var_mean', 'fitness_mean']].copy()
    export_df = export_df.rename(columns={
        'avg_distance_mean': 'Average_Latency',
        'workload_var_mean': 'Workload_Variance',
        'fitness_mean': 'Fitness_Score'
    })
    export_df.to_csv(filename, index=False)
    print(f"Averages exported to {filename} in linear scale")


def generate_scenario7_8_errorbar_plots(data_dir='results/csv'):
    pattern7 = os.path.join(data_dir, 'skenario7_*.csv')
    pattern8 = os.path.join(data_dir, 'skenario8_*.csv')

    df7 = read_scenario_files(pattern7, 'Without PU')
    df8 = read_scenario_files(pattern8, 'With PU')
    df_combined = pd.concat([df7, df8], ignore_index=True)

    if df_combined.empty:
        raise ValueError('No scenario 7 or 8 files were found.')

    summary_df = summarize_combined_data(df_combined)

    print('Generating latency error-bar plot...')
    save_errorbar_plot(
        summary_df,
        'avg_distance_mean',
        'avg_distance_std',
        'Average Latency vs Edge Servers',
        'Average Latency',
        'latency_errorbar.png',
        use_log=False
    )

    print('Generating workload error-bar plot...')
    save_errorbar_plot(
        summary_df,
        'workload_var_mean',
        'workload_var_std',
        'Workload Variance vs Edge Servers',
        'Workload Variance',
        'workload_errorbar.png',
        use_log=True
    )

    print('Generating fitness error-bar plot...')
    save_errorbar_plot(
        summary_df,
        'fitness_mean',
        'fitness_std',
        'Fitness Score vs Edge Servers',
        'Fitness Score',
        'fitness_errorbar.png',
        use_log=True
    )

    print('Exporting averages to CSV...')
    save_averages_to_csv(summary_df)


if __name__ == '__main__':
    generate_scenario7_8_errorbar_plots()
