#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import seaborn as sns

# Function to check if two rectangles intersect or are within a margin
def intersect(rect1, rect2, margin):
    x1, y1, x1_size, y1_size = rect1
    x2, y2, x2_size, y2_size = rect2
    return x1 < x2 + x2_size + margin and x2 < x1 + x1_size + margin and y1 < y2 + y2_size + margin and y2 < y1 + y1_size + margin

# Function to calculate density based on region characteristics
def calculate_density(region_id, x_size, y_size, density_factors):
    density_factor = np.random.uniform(density_factors[region_id][0], density_factors[region_id][1])
    return int((x_size * y_size) / density_factor)

# Function to generate cells for a region
def generate_cells(x_start, y_start, x_size, y_size, n_points):
    x = np.random.uniform(x_start, x_start + x_size, n_points)
    y = np.random.uniform(y_start, y_start + y_size, n_points)
    return x, y

# Main function to simulate regions and cells
def simulate(num_simulations=150, 
             min_region_size=50, max_region_size=150, 
             simulation_area_size=1000, min_noise_points=500, max_noise_points=1500, 
             fixed_margin=20, num_instances_per_region=8, num_region_types=4,
             density_factors={1: (20, 40), 2: (50, 100), 3: (100, 150), 4: (150, 200)},
             output_folder_path=Path('/your_path/data/simulations')):

    os.makedirs(output_folder_path, exist_ok=True)
    max_start = simulation_area_size - max_region_size

    for sim in tqdm(range(num_simulations)):
        noise_points = int(np.random.uniform(min_noise_points, max_noise_points))
        all_x, all_y, all_labels = [], [], []
        existing_regions = []
        
        for instance in range(num_instances_per_region):
            for region_id in range(1, num_region_types + 1):
                retry = True
                while retry:
                    x_start = np.random.uniform(0, max_start)
                    y_start = np.random.uniform(0, max_start)
                    x_size = np.random.uniform(min_region_size, max_region_size)
                    y_size = np.random.uniform(min_region_size, max_region_size)
                    # Check for intersection or proximity within margin with existing regions
                    retry = any(intersect([x_start, y_start, x_size, y_size], er, fixed_margin) for er in existing_regions)
                existing_regions.append([x_start, y_start, x_size, y_size])
                
                # Calculate density based on region characteristics
                density = calculate_density(region_id, x_size, y_size, density_factors)
                x, y = generate_cells(x_start, y_start, x_size, y_size, density)
                
                labels = [f"Region{region_id}"] * len(x)
                all_x.extend(x)
                all_y.extend(y)
                all_labels.extend(labels)
        
        # Generate noise
        noise_x, noise_y = generate_cells(0, 0, simulation_area_size, simulation_area_size, noise_points)
        noise_labels = ["Noise"] * len(noise_x)
        
        all_x.extend(noise_x)
        all_y.extend(noise_y)
        all_labels.extend(noise_labels)
        
        exp_name = f"sim_{sim + 1}"
        exp_path = os.path.join(output_folder_path, exp_name)
        os.makedirs(exp_path, exist_ok=True)
        df = pd.DataFrame({'x': all_x, 'y': all_y, 'region_label': all_labels})
        
        # Save DataFrame
        df.to_csv(os.path.join(exp_path, f"sim_{sim + 1}.csv"), index=False)
        
        # Plot
        plt.figure()
        sns.scatterplot(data=df, x='x', y='y', hue='region_label', palette='tab10', s=10)
        plt.legend(title='Region Label', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(f"Simulation {sim + 1}")

        # Set x-axis to start from 0
        plt.xlim(left=0)

        # Invert the y-axis
        plt.gca().invert_yaxis()

        # Save the plot
        plt.savefig(os.path.join(exp_path, f"figure_{sim + 1}.png"), bbox_inches="tight")
        # plt.show()
        plt.close()
simulate()
# %%
