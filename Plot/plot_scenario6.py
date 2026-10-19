import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 1. Configuration
DATA_DIR = './results/csv' 

def main():
    # Target all 10 files for scenario 6 dynamically
    file_pattern = os.path.join(DATA_DIR, 'skenario6_*.csv')
    all_files = glob.glob(file_pattern)

    if not all_files:
        print(f"No CSV files found matching {file_pattern}. Please check your directory.")
        return

    # 2. Data Extraction and Merging
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all 10 trials into one massive DataFrame
    full_df = pd.concat(df_list, ignore_index=True)

    # 3. Calculate Mean and Standard Deviation across the 10 trials
    # Group by the X-axis (num_edge_servers) and the line separation variable (filter)
    grouped = full_df.groupby(['num_edge_servers', 'filter']).agg(
        latency_mean=('Average', 'mean'),
        latency_std=('Average', 'std'),
        workload_mean=('workload_var', 'mean'),
        workload_std=('workload_var', 'std'),
        distance_mean=('avg_distance', 'mean'),
        distance_std=('avg_distance', 'std')
    ).reset_index()

    # 4. Plotting the 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten axes for targeted placement mapping to your image
    ax1 = axes[0, 0]  # Top-Left: Workload
    ax2 = axes[0, 1]  # Top-Right: Latency
    ax3 = axes[1, 0]  # Bottom-Left: Fitness
    ax4 = axes[1, 1]  # Bottom-Right: Empty

    # Hide the bottom-right graph completely
    ax4.axis('off')
    
    filters = grouped['filter'].unique()
    
    # --- Top-Left Graph: Workload ---
    for f in filters:
        subset = grouped[grouped['filter'] == f]
        ax1.errorbar(subset['num_edge_servers'], subset['workload_mean'], yerr=subset['workload_std'],
                     label=f'Filter {f}', marker='o', capsize=5, capthick=1.5, linewidth=2)
    ax1.set_ylabel('Workload', fontsize=12)
    ax1.set_xlabel('Number of Edge Servers', fontsize=12)
    ax1.set_title('Average Workload vs Number of Edge Servers', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title="Filter Setting")

    # --- Top-Right Graph: Latency ---
    for f in filters:
        subset = grouped[grouped['filter'] == f]
        ax2.errorbar(subset['num_edge_servers'], subset['latency_mean'], yerr=subset['latency_std'],
                     label=f'Filter {f}', marker='o', capsize=5, capthick=1.5, linewidth=2)
    ax2.set_ylabel('Latency', fontsize=12)
    ax2.set_xlabel('Number of Edge Servers', fontsize=12)
    ax2.set_title('Average Latency vs Number of Edge Servers', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Bottom-Left Graph: Fitness ---
    for f in filters:
        subset = grouped[grouped['filter'] == f]
        ax3.errorbar(subset['num_edge_servers'], subset['distance_mean'], yerr=subset['distance_std'],
                     label=f'Filter {f}', marker='o', capsize=5, capthick=1.5, linewidth=2)
    ax3.set_ylabel('Fitness', fontsize=12)
    ax3.set_xlabel('Number of Edge Servers', fontsize=12)
    ax3.set_title('Average Fitness vs Number of Edge Servers', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Main overarching title
    fig.suptitle('Performance Metrics vs Number of Edge Servers', fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 5. Save the combined graph
    save_filename = os.path.join(DATA_DIR, 'scenario6_combined_2x2.png')
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    print(f"Plot successfully generated and saved to: {save_filename}")
    
    plt.close()

if __name__ == "__main__":
    main()