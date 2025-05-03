import utils as ut
from hiergp.hiergp import hierGP, dataManager
import os
import matplotlib.pyplot as plt

# Tasks to perform
generate = True
plot_dists = False
plot_samples = True
map_samples, map_level = True, 7
reduce = True
print_info = True
images_dir = 'images'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Sample paths to generate to and/or read from
paths = ['data/samples1.txt', 'data/samples2.txt', 'data/samples3.txt']

# Sample Generation Configuration
counts_to_generate = [100, 100, 9800]

#           [(name of variable, min, max), ...]
variables = [('var1', 0, 50), ('var1', 0, 50), ('var1', 0, 50)]
sigma_limits = [0.2]

# Create numpy array samples and write to text file
if generate:
    for i, path in enumerate(paths):
        samples = ut.create_samples(counts_to_generate[i], variables[i][0], variables[i][1], variables[i][2])
        if plot_samples:
            ut.plot(samples, images_dir)
        ut.write_to_file(samples, path)


# Load data into Pandas DataFrame from written text file
samples_by_file_list = []
for path in paths:
    samples = ut.read_from_file(path)
    samples_by_file_list.append(samples)

# Plot the distributions of the loaded samples by file
if plot_dists:
    for i, sample_df in enumerate(samples_by_file_list):
        ut.plot_value_distributions(sample_df, paths[i], print_info)

# Determine sample to become (0, 0) and centralize the model instance on (reduces round-off error)
midpoint_sample_by_file_list = []
for i, file_samples in enumerate(samples_by_file_list):
    file_midpoint = ut.get_midpoint_sample(file_samples, print_info)
    midpoint_sample_by_file_list.append(file_midpoint)

# Create an instance of the manager model of specified configuration
manager = dataManager(grid_base_size=25, sigma_limits=sigma_limits, images_dir=images_dir)

# Ingest first sample batch into manager model instance
manager.feed_samples(samples_by_file_list[0], print_info)
manager.feed_samples(samples_by_file_list[1], print_info)
manager.feed_samples(samples_by_file_list[2], print_info)


# Map the samples through levels up to 'map_level'
if map_samples:
    manager.map_samples(map_level, print_info)


# Perform reduction to specified level (doubling base grid size each level)
if reduce:
    manager.reduce_samples(print_info)

    # Plot after mapping and reducing
    if plot_samples:
        manager.plot_samples(print_info)
