import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_results_folder():
    # Create results folder if it doesn't exist
    if not os.path.exists('result_plots'):
        os.makedirs('result_plots')

def analyze_benchmark_extended(csv_file):
    create_results_folder()
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate metrics grouped by algorithm, set type, and size
    metrics = {
        'avg': df.groupby(['Algorithm', 'Set_Type', 'Size_Category'])['Best_Fitness'].mean().reset_index(),
        'max': df.groupby(['Algorithm', 'Set_Type', 'Size_Category'])['Best_Fitness'].max().reset_index(),
        'min': df.groupby(['Algorithm', 'Set_Type', 'Size_Category'])['Best_Fitness'].min().reset_index()
    }
    
    # Pivot data for plotting
    plot_data = {
        metric: data.pivot_table(
            index=['Set_Type', 'Size_Category'],
            columns='Algorithm',
            values='Best_Fitness'
        ).reset_index()
        for metric, data in metrics.items()
    }
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 24))
    titles = {
        'avg': 'Average Performance',
        'max': 'Maximum Performance',
        'min': 'Minimum Performance'
    }
    
    colors = {
        'Baseline': '#8884d8',
        'SimulatedAnnealing': '#82ca9d',
        'GeneticAlgorithm': '#ffc658',
        'AntColony': '#ff7300'
    }
    
    for (metric, data), ax in zip(plot_data.items(), axes):
        bar_width = 0.2
        x = range(len(data))
        
        for i, (algo, color) in enumerate(colors.items()):
            ax.bar(
                [xi + (i-1.5)*bar_width for xi in x], 
                data[algo], 
                bar_width, 
                label=algo, 
                color=color
            )
        
        ax.set_xlabel('Instance Category')
        ax.set_ylabel('Fitness Score')
        ax.set_title(f'{titles[metric]} by Instance Category')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['Set_Type']} {row['Size_Category']}" for _, row in data.iterrows()],
            rotation=45
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result_plots/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create box plot
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=df,
        x='Set_Type',
        y='Best_Fitness',
        hue='Algorithm',
        dodge=True
    )
    plt.xticks(rotation=45)
    plt.title('Performance Distribution by Algorithm and Instance Type')
    plt.tight_layout()
    plt.savefig('result_plots/performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return plot_data

def plot_routes_performance(csv_file):
    create_results_folder()
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create subplots for each size category and set type combination
    size_categories = df['Size_Category'].unique()
    set_types = df['Set_Type'].unique()
    
    fig, axes = plt.subplots(len(size_categories), len(set_types), 
                            figsize=(15*len(set_types), 6*len(size_categories)))
    
    colors = {
        'Baseline': '#8884d8',
        'SimulatedAnnealing': '#82ca9d',
        'GeneticAlgorithm': '#ffc658',
        'AntColony': '#ff7300'
    }
    
    for i, size in enumerate(size_categories):
        for j, set_type in enumerate(set_types):
            ax = axes[i][j] if len(size_categories) > 1 and len(set_types) > 1 else axes
            instance_data = df[(df['Size_Category'] == size) & (df['Set_Type'] == set_type)]
            
            for algorithm in colors.keys():
                algo_data = instance_data[instance_data['Algorithm'] == algorithm]
                algo_data = algo_data.sort_values(['Number_of_Routes', 'Best_Fitness'])
                
                ax.scatter(
                    algo_data['Number_of_Routes'],
                    algo_data['Best_Fitness'],
                    label=algorithm,
                    color=colors[algorithm],
                    alpha=0.7,
                    s=100
                )
                
                if len(algo_data) > 1:
                    ax.plot(
                        algo_data['Number_of_Routes'],
                        algo_data['Best_Fitness'],
                        color=colors[algorithm],
                        alpha=0.3
                    )
            
            ax.set_title(f'Performance vs Routes - {size} {set_type} instances')
            ax.set_xlabel('Number of Routes')
            ax.set_ylabel('Best Fitness')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('result_plots/routes_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate detailed statistics and save to CSV
    route_stats = df.groupby(['Set_Type', 'Size_Category', 'Algorithm'])[
        ['Number_of_Routes', 'Best_Fitness']
    ].agg({
        'Number_of_Routes': ['mean', 'min', 'max'],
        'Best_Fitness': ['mean', 'min', 'max']
    }).round(2)
    
    # Save statistics to CSV
    route_stats.to_csv('result_plots/route_statistics.csv')
    print("\nRoutes Analysis saved to 'result_plots/route_statistics.csv'")
    print(route_stats)

# Example usage:
analyze_benchmark_extended('benchmark_results_20241119_225823/complete_benchmark_summary.csv')
plot_routes_performance('benchmark_results_20241119_225823/complete_benchmark_summary.csv')