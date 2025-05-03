import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

# Set the random seed for reproducibility
np.random.seed(42)


# Generate synthetic latitude and longitude samples
def create_samples(n, var_name="temperature", var_min=0, var_max=30):
    lat_min, lat_max = 46.7, 46.8  # Latitude range
    lon_min, lon_max = -117.2, -117.1  # Longitude range
    lat = np.random.uniform(lat_min, lat_max, n)  # Latitude samples
    lon = np.random.uniform(lon_min, lon_max, n)  # Longitude samples

    # Generate a variable with a heat map over it
    temp = np.zeros(n)  # Initialize value samples

    # Assign values based on the distance to a reference point
    # The closer the point is to the reference point, the higher the value
    ref_lat, ref_lon = 46.75, -117.15  # Reference point
    
    # Calculate Euclidean distance
    dist = np.sqrt((lat - ref_lat)**2 + (lon - ref_lon)**2)

    # Assign value based on distance
    temp = var_max - dist * (var_max - var_min)

    # Create a pandas dataframe with the samples
    return pd.DataFrame({'lat': lat, 'long': lon, var_name: temp})


def plot(samples, images_dir=""):
    var_name = samples.columns[2]
    # Plot the points with color map
    plt.scatter(x=samples['long'], y=samples['lat'], c=samples[var_name], cmap='jet')
    plt.colorbar(label=var_name)  # Add a color bar
    plt.xlabel('Longitude')  # Add x-axis label
    plt.ylabel('Latitude')  # Add y-axis label
    plt.title('Synthetic Data Heat Map')  # Add title
    if not (images_dir == ""):
        plt.savefig(join(images_dir, 'original.png'))
    plt.show()  # Show the plot

def write_to_file(samples, file_name):
    # Write to file with tab separator and no index
    samples.to_csv(file_name, sep='\t', index=False)


def read_from_file(file_name):
    # Read from file with tab separator
    return pd.read_csv(file_name, sep='\t')


def plot_value_distributions(sample_df, path, print_info=False):
    n, m = sample_df.shape
    if print_info:
        print("_" * 100)
        print(f"Plotting samples from path = {path} ({n} samples with {m} columns)")

    # Create subplots
    fig, axs = plt.subplots(m)
    fig.suptitle(f"Samples from '{path}'")

    # Generate set of m unique colors
    cmap = plt.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, m))

    # Open the file
    var_names = sample_df.columns.tolist()

    # Plot histograms
    bins = floor(sqrt(n))
    samples = sample_df.to_numpy()
    for i, var in enumerate(range(m)):
        values = samples[:, i]
        axs[i].hist(values, bins=bins, color=colors[i], alpha=0.7)
        q1, q2, q3 = np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75)
        axs[i].axvline(q1, color='r', linestyle='dashed', linewidth=2)
        axs[i].axvline(q2, color='g', linestyle='dashed', linewidth=2)
        axs[i].axvline(q3, color='b', linestyle='dashed', linewidth=2)
        axs[i].set_title(f'{var_names[i]} (Q1 = {q1:.2f}, Q2 = {q2:.2f}, Q3 = {q3:.2f})')

    # Show the plot
    plt.tight_layout()
    plt.show()


def get_midpoint_sample(sample_df, print_info=False):
    samples = sample_df.to_numpy()
    lat_avg, long_avg = np.mean(samples[:, 0]), np.mean(samples[:, 1])
    diffs_lat, diffs_long = np.abs(samples[:, 0] - lat_avg), np.abs(samples[:, 1] - long_avg)
    sum_diffs = diffs_lat + diffs_long
    index_min_diff_sample = np.argmin(sum_diffs)
    midpoint_sample = samples[index_min_diff_sample]
    if print_info:
        print("_" * 100)
        print(f"Averages: lat = {lat_avg}, long = {long_avg}")
        print(f"Best midpoint sample is index {index_min_diff_sample} = {midpoint_sample}")
    return midpoint_sample[0], midpoint_sample[1]