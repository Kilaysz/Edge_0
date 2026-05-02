import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_nametag_plots(file7, file8):
    # 1. Load the data
    df7 = pd.read_csv(file7)
    df8 = pd.read_csv(file8)

    # 2. Add the "Nametags" (The Scenarios)
    df7['Scenario'] = 'Without PU'
    df8['Scenario'] = 'With PU'

    # 3. Combine them into one big table
    df_combined = pd.concat([df7, df8], ignore_index=True)

    # Set up a distinct color palette
    algorithms = df_combined['placer_name'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    palette = dict(zip(algorithms, colors))

    # Helper function to generate single plots
    def save_metric_plot(y_col, title, ylabel, filename, use_log=False):
        plt.figure(figsize=(12, 7)) # Larger canvas gives lines room to breathe
        
        # 4. Split by Line Style (style='Scenario')
        sns.lineplot(
            data=df_combined,
            x='num_edge_servers',
            y=y_col,
            hue='placer_name',     # Color represents the Algorithm
            style='Scenario',      # Line Style represents the Scenario (Solid vs Dashed)
            markers=True,
            dashes=True,           # Enables dashed lines for the second scenario
            palette=palette,
            linewidth=2.5,
            markersize=8
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Number of Edge Servers', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Apply Log Scale if requested (crucial for preventing crowded workload lines)
        if use_log:
            plt.yscale('log')
            
        # Place the legend completely outside the plot
        plt.legend(
            title='Algorithms & Scenarios', 
            fontsize='11', 
            title_fontsize='13',
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Generate the 3 separate PNG files
    print("Generating latency plot...")
    save_metric_plot('avg_distance', 'Average Latency vs Edge Servers', 'Average Latency (Distance)', 'latency_nametag.png')
    
    print("Generating workload plot...")
    save_metric_plot('workload_var', 'Workload Variance vs Edge Servers', 'Workload Variance (Log Scale)', 'workload_nametag.png', use_log=True)
    
    print("Generating fitness plot...")
    save_metric_plot('fitness', 'Fitness Score vs Edge Servers', 'Fitness Score (Log Scale)', 'fitness_nametag.png', use_log=True)
    print("Done!")

# Run the script
if __name__ == "__main__":
    generate_nametag_plots('results\\csv\\skenario7.csv', 'results\\csv\\skenario8.csv')
