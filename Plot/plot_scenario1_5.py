import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Configuration
# Set the path to the folder containing your CSV files ('.' means the current folder)
DATA_DIR = './results/csv' 

# Define settings for each scenario, including the specific filename to save
scenario_configs = {
    1: {'title': 'Scenario 1: Effect of Iterations', 'x_label': 'Number of Iterations', 'color': '#1f77b4', 'filename': 'scenario1_iterations.png'},
    2: {'title': 'Scenario 2: Effect of Swarm Size', 'x_label': 'Swarm Size (M)', 'color': '#ff7f0e', 'filename': 'scenario2_swarmsize.png'},
    3: {'title': 'Scenario 3: Effect of Beta (β)', 'x_label': 'Contraction Expansion Coefficient (β)', 'color': '#2ca02c', 'filename': 'scenario3_beta.png'},
    4: {'title': 'Scenario 4: Effect of Mu (μ)', 'x_label': 'Weight Parameter (μ)', 'color': '#d62728', 'filename': 'scenario4_mu.png'},
    5: {'title': 'Scenario 5: Effect of Server Count', 'x_label': 'Number of Servers (K)', 'color': '#9467bd', 'filename': 'scenario5_servercount.png'}
}

# 2. Loop through the Scenarios individually 
for scenario_num in range(1, 6):
    # Target the single file for this specific scenario (e.g., skenario1.csv)
    file_path = os.path.join(DATA_DIR, f'skenario{scenario_num}.csv')
    
    if not os.path.exists(file_path):
        print(f"Skipping Scenario {scenario_num}: File not found at {file_path}")
        continue

    # 3. Data Extraction
    try:
        df = pd.read_csv(file_path)
        
        # The first column is assumed to be the independent parameter (e.g., 'mu', 'Iterations')
        x_col_name = df.columns[0] 
        
        if 'Average' not in df.columns:
             print(f"Warning: 'Average' column not found in {file_path}")
             continue

        # Automatically calculate Standard Deviation based on Trial columns
        # (Assuming columns are named like 'Trial_1', 'Trial_2', etc.)
        trial_cols = [col for col in df.columns if 'trial' in col.lower()]
        
        if len(trial_cols) > 1:
            df['std_dev'] = df[trial_cols].std(axis=1)
        elif 'std' in df.columns or 'Std' in df.columns:
            # Fallback in case you already have a pre-calculated 'std' column
            std_col = 'std' if 'std' in df.columns else 'Std'
            df['std_dev'] = df[std_col]
        else:
            # If no trials or std column found, set error to 0 to prevent crashes
            df['std_dev'] = 0
            print(f"Notice: No multiple 'Trial' columns found in {file_path}. Error bars will be 0.")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    # 4. Plotting a Single Separate Graph
    config = scenario_configs[scenario_num]
    
    plt.figure(figsize=(8, 6)) # Create a fresh canvas for each scenario
    
    # Plotting using errorbar instead of standard plot
    plt.errorbar(df[x_col_name], df['Average'], yerr=df['std_dev'],
                 marker='o', linestyle='-', color=config['color'], 
                 markersize=6, linewidth=2, capsize=5, capthick=1.5,
                 label='Average Fitness')

    # Formatting the Graph
    plt.title(config['title'], fontsize=14, fontweight='bold')
    plt.xlabel(config['x_label'], fontsize=12)
    plt.ylabel('Combined Fitness Value', fontsize=12)
    
    plt.xticks(df[x_col_name]) # Ensure correct X-axis intervals
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    
    # 5. Save the separate image file
    plt.savefig(config['filename'], dpi=300)
    print(f"Successfully generated {config['filename']}")
    
    # Close the figure so the next scenario starts completely blank
    plt.close() 

print("All graphs have been generated successfully!")