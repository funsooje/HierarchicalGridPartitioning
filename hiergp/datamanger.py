import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from hiergp import hierGP

# HierGP Data Manager Class
class dataManager: 

    def __init__(self, grid_base_size, sigma_limits, images_dir):
        
        # Initialize hierGP
        self.grid_base_size = grid_base_size
        self.hgp = hierGP(base_size = self.grid_base_size)

        self.images_dir = images_dir

        # Initialize samples to empty Pandas Dataframe
        self.samples = pd.DataFrame()

        # Initialize schema to None, used for easily grabbing samples in form with initial schema
        self.schema = None

        # Midpoint to offset each sample for grid indexing
        self.midpoint = None

        # Set the Kolmogorov-Smirnov test acceptable p-value for reduction
        self.sigma_limits = sigma_limits

        self.highest_reduction_level = 0

    def feed_samples(self, sample_df, print_info=False):
        if print_info:
            print("-" * 100)
            print(f"Adding {sample_df.shape[0]} samples...")

        # No prior samples, set up schema of the model's samples moving forward
        if self.samples.empty:
            self.schema = sample_df.columns.tolist()
            self.midpoint = self._get_midpoint_sample(sample_df, print_info)
            if print_info:
                print(f"NOTE: First mapping detected.\n\tRequired schema moving forward = {self.schema}.")
            # Set samples to input sample set with 'centers'
            self.samples = self._add_centered_lat_long(sample_df)
        else:
            if sample_df.columns.tolist() == self.schema:
                all_samples = pd.concat([self.samples[self.schema], sample_df])
                # Check that new samples being adding conform to schema of samples added previously
                new_midpoint = self._get_midpoint_sample(all_samples, print_info)
                if new_midpoint != self.midpoint:
                    self.midpoint = new_midpoint
                    if print_info:
                        print(f"New midpoint found and assigned.")
                self.samples = self._add_centered_lat_long(all_samples)
            # Mismatch columns, cannot add to samples
            else:
                raise ValueError("Error: adding samples with different schema than initial model seeding.")
        if print_info:
            print(f"Successfully added samples. New sample total = {self.samples.shape[0]}.")

    def _add_centered_lat_long(self, samples):
        column_names = samples.columns.values.tolist()
        # Convert samples to numpy for operations
        samples = samples.to_numpy()

        # Create new columns for offset lat and long values ('lat_cent' & 'long_cent')
        samples = np.hstack((samples, samples[:, :2]))

        # Get all points' lat and long values offset to "midpoint" value as center point (0, 0)
        samples[:, -2] -= self.midpoint[0]
        samples[:, -1] -= self.midpoint[1]
        column_names = column_names + ['lat_cent', 'long_cent']

        # Replace samples instance field with adjusted version
        return pd.DataFrame(samples, columns=column_names)

    def _get_midpoint_sample(self, sample_df, print_info=False):
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

    def map_samples(self, highest_level, print_info=False):
        samples = pd.DataFrame()

        if print_info:
            print("_" * 100)
            print(f"Mapping {len(self.samples)} samples up to level {highest_level}...")

        col_names = self.samples.columns
        lat_name = col_names[-2]
        lon_name = col_names[-1]
        for i, reduction_level in enumerate(range(1, highest_level + 1)):
            temp = self.hgp.generateGrids(self.samples, reduction_level, lat=lat_name, lon=lon_name)
            selected_columns = temp.iloc[:, -4:]
            rearranged_columns = selected_columns.iloc[:, [2, 3, 0, 1]]
            rearranged_columns.columns = [f'X{i + 1}', f'Y{i + 1}', f'Xmid{i + 1}', f'Ymid{i + 1}'] #['X1', 'Y2', 'Xmid1', 'Ymid2']
            samples = pd.concat([samples, rearranged_columns], axis=1)

        # Record highest reduction level for plotting convenience
        self.highest_reduction_level = highest_level

        self.samples = pd.concat([self.samples, samples], axis=1)

        if print_info:
            print("Done!")

    def find_centroid(self, samples):
        return samples['long_cent'].mean(), samples['lat_cent'].mean(),

    def reduce_samples(self, print_info):
        # Create a copy of samples so don't accidentally alter the original
        samples_df = self.samples.copy()

        if print_info:
            print("_" * 100)
            print(f"Starting reduction process...")

        # Get the values from highest to lowest reduction level and test in passes test to reduce
        for i, reduction_level in enumerate(reversed(range(self.highest_reduction_level))):
            if print_info:
                print(f"Level {reduction_level}")
            # Get the grid X, Y index values for this level
            x = samples_df['X' + str(reduction_level + 1)].to_numpy()
            y = samples_df['Y' + str(reduction_level + 1)].to_numpy()

            # Get the range of values for both X and Y index values
            min_x_index, max_x_index = np.min(x), np.max(x)
            min_y_index, max_y_index = np.min(y), np.max(y)

            # For each possible X index in its possible range
            for x_index in np.arange(min_x_index, max_x_index + 1, 1):
                # For each possible Y index in its possible range
                for y_index in np.arange(min_y_index, max_y_index + 1, 1):
                    # Get all the samples that are assigned to this grid cell at this level
                    matching_samples = self.samples[(self.samples['X' + str(reduction_level + 1)] == x_index)
                                                    & (self.samples['Y' + str(reduction_level + 1)] == y_index)]
                    reduce = True
                    if matching_samples.empty:
                        reduce = False
                    elif len(matching_samples) > 1:
                        for j in range(len(self.schema) - 2):
                            var_value_variance = matching_samples.iloc[:, j + 2].std(axis=0)
                            variance_limit = self.sigma_limits[j]
                            if var_value_variance > variance_limit:
                                reduce = False
                            if print_info:
                                if reduce:
                                    print(f"\t({x_index}, {y_index}) PASS: var{j} sigma {var_value_variance} <= limit "
                                          f"of {variance_limit}.")
                                else:
                                    print(f"\t({x_index}, {y_index}) FAIL: var{j} sigma {var_value_variance} > limit "
                                          f"of {variance_limit}.")
                    else:
                        if print_info:
                            print(f"\t({x_index}, {y_index}) PASS: only one sample, sigma(s) UNDEFINED.")

                    # One matching sample auto passes limits
                    if reduce:
                        long_mid, lat_mid = self.find_centroid(matching_samples)
                        matching_samples.loc[:, 'Xmid' + str(reduction_level + 1)] = long_mid
                        matching_samples.loc[:, 'Ymid' + str(reduction_level + 1)] = lat_mid
                    else:
                        self.samples['Xmid' + str(reduction_level + 1)] = matching_samples[
                            'Xmid' + str(reduction_level + 1)].astype(object)
                        matching_samples.loc[:, 'Xmid' + str(reduction_level + 1)] = np.nan
                        self.samples['Ymid' + str(reduction_level + 1)] = matching_samples[
                            'Ymid' + str(reduction_level + 1)].astype(object)
                        matching_samples.loc[:, 'Ymid' + str(reduction_level + 1)] = np.nan
                    self.samples.update(matching_samples)

    def print_model_info(self):
        print("-" * 100)
        print("MODEL INFO:")
        # TODO: populate this


    def plot_samples(self, print_info=False):
        # Create a copy of samples so don't accidentally alter the original
        samples_df = self.samples.copy()

        if print_info:
            print("_" * 100)
            print("Plotting samples...")

        # Get number of samples as 'n' and number of columns as 'm'
        n, m = self.samples.shape

        # If no samples to plot, raise error
        if n == 0:
            raise ValueError("Error: can not plot empty set of samples.")

        # Generate set of m unique colors (one for each sample for entirely unmapped case)
        cmap = plt.colormaps.get_cmap('turbo')
        colors = cmap(np.linspace(0, 1, n))
        np.random.shuffle(colors)  # shuffle so each color is placed next to colors more likely distinct to it
        color_cycle = itertools.cycle(colors)

        # Get relative unmapped "true" lat and long values
        lats, longs = self.samples['lat_cent'].to_numpy(), self.samples['long_cent'].to_numpy()

        # Determine whether any mapping has been performed to plot
        if m == len(self.schema) + 2:  # Unmapped = ['schema'] + ['lat_cent', 'long_cent']
            for i, long in enumerate(longs):
                plt.plot(longs[i], lats[i], 'o', color=colors[i], markersize=2)
            plt.title(f"{self.samples.shape[0]} unmapped samples")

        else:
            reduction_results = pd.DataFrame(columns=['Xmid', 'Ymid', 'avg'])
            colors_locked, color_index = [], -1
            # Get the values from highest to lowest reduction level and plot
            for reduction_level in reversed(range(self.highest_reduction_level)):
                if print_info:
                    print(f"Level {reduction_level + 1}")

                # Get the grid X, Y index values for this level
                x = self.samples['X' + str(reduction_level + 1)].to_numpy()
                y = self.samples['Y' + str(reduction_level + 1)].to_numpy()

                # Get the range of values for both X and Y index values
                min_x_index, max_x_index = np.min(x), np.max(x)
                min_y_index, max_y_index = np.min(y), np.max(y)

                # For each possible X index in its possible range
                for x_index in np.arange(min_x_index, max_x_index + 1, 1):
                    # For each possible Y index in its possible range
                    for y_index in np.arange(min_y_index, max_y_index + 1, 1):
                        # Get all the samples that are assigned to this grid cell at this level
                        matching_samples = samples_df[(samples_df['X' + str(reduction_level + 1)] == x_index)
                                                      & (samples_df['Y' + str(reduction_level + 1)] == y_index)]

                        # Deselect any not reducible to this level
                        not_nan_mask = matching_samples['Xmid' + str(reduction_level + 1)].notna()
                        reduced_samples = matching_samples[not_nan_mask]

                        # If at least a sample is indexed to this grid cell
                        if not reduced_samples.empty:
                            if print_info:
                                print(f"\t({x_index}, {y_index}): {len(reduced_samples)} reducible samples = ")
                                print("first 5 =")
                                print(reduced_samples.head(5))

                            # Get the centered long and lat values for the matching samples
                            match_longs, match_lats = (reduced_samples['long_cent'].to_numpy(),
                                                       reduced_samples['lat_cent'].to_numpy())

                            # Get the midpoint values for the matching samples
                            x_mid, y_mid = (reduced_samples['Xmid' + str(reduction_level + 1)].to_numpy()[0],
                                            reduced_samples['Ymid' + str(reduction_level + 1)].to_numpy()[0])

                            avgs_var1 = reduced_samples['var1'].mean()

                            reduction_results.loc[len(reduction_results)] = [x_mid, y_mid, avgs_var1]

                            # Get a unique color for this grid cell
                            color = next(color_cycle)

                            # Plot the points
                            plt.plot(match_longs, match_lats, 's', color=color, markersize=4, zorder=2)
                            plt.plot(x_mid, y_mid, '*', color=color, markersize=10 * (reduction_level+1),
                                     zorder=2, markeredgewidth=1, markeredgecolor='black')

                            # Remove the points from the pool of other samples left to plot
                            samples_df = samples_df.drop(reduced_samples.index)

                # Stop early if all samples already plotted
                if samples_df.empty:
                    break

            # Get the global minimum and maximum
            vmin = min(reduction_results['avg'].min(), self.samples['var1'].min())
            vmax = max(reduction_results['avg'].max(), self.samples['var1'].max())

            # Plot the first scatter plot
            plt.scatter(x=reduction_results['Xmid'], y=reduction_results['Ymid'],
                        c=reduction_results['avg'], cmap='jet', vmin=vmin, vmax=vmax,
                        marker='*', s=15, facecolors='none', zorder=3)

            # Plot the second scatter plot
            plt.scatter(x=self.samples['long_cent'], y=self.samples['lat_cent'],
                        c=self.samples['var1'], cmap='jet', vmin=vmin, vmax=vmax,
                        marker='.', s=4, facecolors='none', zorder=3)

            # Add title to plot
            plt.title(f"{self.samples.shape[0]} mapped samples by grid assignments up to level "
                      f"{self.highest_reduction_level}")
        # Compute positions of vertical and horizontal grid lines on plot
        linewidth = 0.1  # have all grid lines the same width
        # Get the boundary values
        buffer = self.delta_lat
        leftmost_long_value, rightmost_long_value = abs(min(longs)) + buffer, max(longs) + buffer
        bottommost_lat_value, topmost_lat_value = abs(min(lats)) + buffer, max(lats) + buffer

        # Create the array 0 to boundary for negative values and then reverse ordering for concatenation with positives
        min_range_long = np.arange(0, leftmost_long_value, self.delta_long)[::-1] * -1
        min_range_lat = np.arange(0, bottommost_lat_value, self.delta_long)[::-1] * -1

        # Drop the last value of 0 from each negative list so only 1 appears in final tick positions lists
        min_range_long, min_range_lat = min_range_long[:-1], min_range_lat[:-1]

        # Create the array 0 to boundary for positive values
        max_range_long = np.arange(0, rightmost_long_value, self.delta_long)
        max_range_lat = np.arange(0, topmost_lat_value, self.delta_long)

        # Concatenate negative and positive lists into one
        long_range = np.concatenate((min_range_long, max_range_long))
        lat_range = np.concatenate((min_range_lat, max_range_lat))

        # Add vertical lines (separated from horizontal for loop in case different numbers to plot)
        for x in long_range:
            plt.axvline(x=x, color='black', linestyle='-', linewidth=linewidth)
        # Add horizontal lines
        for y in lat_range:
            plt.axhline(y=y, color='black', linestyle='-', linewidth=linewidth)

        # Add labels
        plt.xlabel('Centered longitude')
        plt.ylabel('Centered latitude')

        plt.savefig(self.images_dir + '/reductions.png')
        plt.show()
        if print_info:
            print("Done.")
