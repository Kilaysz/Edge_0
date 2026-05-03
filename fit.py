import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 1. Configuration
# Set the path to the folder containing your 40 CSV files ('.' means the current folder)
DATA_DIR = './results/csv' 

# Define settings for each scenario, including the specific filename to save
scenario_configs = {
    1: {'title': 'Scenario 1: Effect of Iterations', 'x_label': 'Number of Iterations', 'color': '#1f77b4', 'filename': 'scenario1_iterations.png'},
    2: {'title': 'Scenario 2: Effect of Swarm Size', 'x_label': 'Swarm Size (M)', 'color': '#ff7f0e', 'filename': 'scenario2_swarmsize.png'},
    3: {'title': 'Scenario 3: Effect of Beta (β)', 'x_label': 'Contraction Expansion Coefficient (β)', 'color': '#2ca02c', 'filename': 'scenario3_beta.png'},
    4: {'title': 'Scenario 4: Effect of Mu (μ)', 'x_label': 'Weight Parameter (μ)', 'color': '#d62728', 'filename': 'scenario4_mu.png'},
    5: {'title': 'Scenario 5: Effect of Server Count', 'x_label': 'Number of Servers (K)', 'color': '#9467bd', 'filename': 'scenario5_servercount.png'}

}

# 2. Loop through the 4 Scenarios individually
for scenario_num in range(5, 6):
    # Find all 10 files for this specific scenario (e.g., skenario1_1.csv to skenario1_10.csv)
    file_pattern = os.path.join(DATA_DIR, f'skenario{scenario_num}_*.csv')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"Skipping Scenario {scenario_num}: No files found matching {file_pattern}")
        continue

    # 3. Data Aggregation
    all_data_frames = []
    x_col_name = None

    for file in file_list:
        try:
            df = pd.read_csv(file)
            
            # The first column is assumed to be the independent parameter (e.g., 'mu', 'Iterations')
            if x_col_name is None:
                x_col_name = df.columns[0] 
            
            # Extract the independent variable and the 'Average' column for each file
            if 'Average' in df.columns:
                df_temp = df[[x_col_name, 'Average']].rename(columns={'Average': 'File_Avg'})
                all_data_frames.append(df_temp)
            else:
                 print(f"Warning: 'Average' column not found in {file}")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data_frames:
         continue

    # 4. Combine data and calculate Grand Averages/Errors
    combined_df = pd.concat(all_data_frames)
    summary_df = combined_df.groupby(x_col_name).agg(
        Grand_Avg=('File_Avg', 'mean'),
        Grand_Std=('File_Avg', 'std')
    ).reset_index()

    # 5. Plotting a Single Separate Graph
    config = scenario_configs[scenario_num]
    
    plt.figure(figsize=(8, 6)) # Create a fresh canvas for each scenario
    
    plt.errorbar(summary_df[x_col_name], summary_df['Grand_Avg'], yerr=summary_df['Grand_Std'], 
                fmt='-o', color=config['color'], ecolor='black', 
                elinewidth=1.5, capsize=5, markersize=6, linewidth=2, 
                label='Grand Average Fitness')

    # Formatting the Graph
    plt.title(config['title'], fontsize=14, fontweight='bold')
    plt.xlabel(config['x_label'], fontsize=12)
    plt.ylabel('Combined Fitness Value', fontsize=12)
    
    plt.xticks(summary_df[x_col_name]) # Ensure correct X-axis intervals
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    
    # 6. Save the separate image file
    plt.savefig(config['filename'], dpi=300)
    print(f"Successfully generated {config['filename']}")
    
    # Close the figure so the next scenario starts completely blank
    plt.close() 

print("All graphs have been generated successfully!")