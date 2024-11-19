import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_benchmark_extended(csv_file):
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
    
    # Colors for algorithms
    colors = {
        'Baseline': '#8884d8',
        'SimulatedAnnealing': '#82ca9d',
        'GeneticAlgorithm': '#ffc658',
        'AntColony': '#ff7300'
    }
    
    for (metric, data), ax in zip(plot_data.items(), axes):
        bar_width = 0.2
        x = range(len(data))
        
        # Plot bars for each algorithm
        for i, (algo, color) in enumerate(colors.items()):
            ax.bar(
                [xi + (i-1.5)*bar_width for xi in x], 
                data[algo], 
                bar_width, 
                label=algo, 
                color=color
            )
        
        # Customize each subplot
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
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return plot_data

def plot_routes_performance(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create subplots for each size category
    size_categories = df['Size_Category'].unique()
    fig, axes = plt.subplots(len(size_categories), 1, figsize=(15, 6*len(size_categories)))
    
    colors = {
        'Baseline': '#8884d8',
        'SimulatedAnnealing': '#82ca9d',
        'GeneticAlgorithm': '#ffc658',
        'AntColony': '#ff7300'
    }
    
    for idx, size in enumerate(size_categories):
        ax = axes[idx] if len(size_categories) > 1 else axes
        size_data = df[df['Size_Category'] == size]
        
        for algorithm in colors.keys():
            algo_data = size_data[size_data['Algorithm'] == algorithm]
            # Sort by number of routes and fitness
            algo_data = algo_data.sort_values(['Number_of_Routes', 'Best_Fitness'])
            
            ax.scatter(
                algo_data['Number_of_Routes'],
                algo_data['Best_Fitness'],
                label=algorithm,
                color=colors[algorithm],
                alpha=0.7,
                s=100
            )
            
            # Add trend line
            if len(algo_data) > 1:
                ax.plot(
                    algo_data['Number_of_Routes'],
                    algo_data['Best_Fitness'],
                    color=colors[algorithm],
                    alpha=0.3
                )
        
        ax.set_title(f'Performance vs Number of Routes - {size} instances')
        ax.set_xlabel('Number of Routes')
        ax.set_ylabel('Best Fitness')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('routes_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics for routes
    print("\nRoutes Analysis:")
    route_stats = df.groupby(['Size_Category', 'Algorithm'])[['Number_of_Routes', 'Best_Fitness']].agg({
        'Number_of_Routes': ['mean', 'min', 'max'],
        'Best_Fitness': ['mean', 'min', 'max']
    }).round(2)
    print(route_stats)


analyze_benchmark_extended('benchmark_results_20241119_215537/complete_benchmark_summary.csv')
plot_routes_performance('benchmark_results_20241119_215537/complete_benchmark_summary.csv')
