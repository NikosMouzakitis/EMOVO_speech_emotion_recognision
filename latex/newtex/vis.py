import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_features_by_gender(csv_path, output_dir="visualizations"):
    """Visualize features against gender from the features CSV"""
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract gender from speaker ID (assuming 'f' for female, 'm' for male)
    df['gender'] = df['speaker'].apply(lambda x: 'female' if x.startswith('f') else 'male')
    
    # List of features to visualize
    features = ['DoC', 'Density', 'CC', 'L', 'M', 'Avg_A']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Feature Distributions by Gender (Window Size: {df["window_size"].iloc[0]})', fontsize=16)
    
    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i//3, i%3]
        
        # Boxplot version
        sns.boxplot(data=df, x='gender', y=feature, ax=ax, palette={'male':'skyblue', 'female':'lightpink'})
        sns.stripplot(data=df, x='gender', y=feature, ax=ax, color='black', alpha=0.5, jitter=True)
        
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('Feature Value')
    
    plt.tight_layout()
    
    # Save and show
    output_path = os.path.join(output_dir, f"gender_comparison_ws{df['window_size'].iloc[0]}.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    plt.show()

# Example usage - run for each window size
visualize_features_by_gender("features/project_emovo_features_1000.csv")
visualize_features_by_gender("features/project_emovo_features_2000.csv") 
visualize_features_by_gender("features/project_emovo_features_3000.csv")
