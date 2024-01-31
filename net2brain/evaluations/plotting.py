import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('ggplot')

class Plotting:
    """Class for plotting the results generated in the evaluation module."""

    def __init__(self, dataframes):
        """Initialization

        Args:
            dataframes (list or DataFrame): List of pandas dataframes or a single dataframe. 
        """
        if isinstance(dataframes, pd.DataFrame):
            self.dataframes = [dataframes]
        elif isinstance(dataframes, list) and all(isinstance(df, pd.DataFrame) for df in dataframes):
            self.dataframes = dataframes
        else:
            raise TypeError("The 'dataframes' argument must be either a DataFrame or a list of DataFrames.")

    def prepare_dataframe(self, dataframe, metric):
        """Prepare the dataframe with the necessary metric calculations."""
        if metric not in dataframe.columns:
            if metric == 'R2':
                dataframe['R2'] = dataframe['R']**2
                dataframe['LNC'] = dataframe['LNC']**2
                dataframe['UNC'] = dataframe['UNC']**2
            elif metric == 'R':
                dataframe['R'] = np.sqrt(dataframe['R2'])
                dataframe['LNC'] = np.sqrt(dataframe['LNC'])
                dataframe['UNC'] = np.sqrt(dataframe['UNC'])
        return dataframe



    def add_significance_markers(self, ax, plotting_df, metric, pairs):
        """Add significance markers based on individual significance."""
        for index, row in plotting_df.iterrows():
            if row['Significance'] < 0.05:
                x = ax.patches[index].get_x() + ax.patches[index].get_width() / 2
                y = ax.patches[index].get_height()
            ax.text(x, y + 0.01, '*', ha='center', va='bottom', c='k')


    def add_noise_ceiling(self, ax, plotting_df):
        """Add noise ceiling lines to the plot."""
        for index, row in plotting_df.iterrows():
            if np.isnan(row['LNC']) or np.isnan(row['UNC']):
                continue
            x = ax.patches[index].get_x()
            width = ax.patches[index].get_width()
            ax.hlines(y=row['LNC'], xmin=x, xmax=x+width, linewidth=1, color='k', linestyle='dashed')
            ax.hlines(y=row['UNC'], xmin=x, xmax=x+width, linewidth=1, color='k', linestyle='dashed')





    def plot(self, pairs=[], metric='R2'):
        max_dataframes = []
        for dataframe in self.dataframes:
            dataframe = self.prepare_dataframe(dataframe, metric)
            max_dataframes.append(dataframe.loc[dataframe.groupby('ROI')[metric].idxmax()])
        
        plotting_df = pd.concat(max_dataframes, ignore_index=True)

        # Extract numerical part from ROI names for sorting
        plotting_df['ROI_num'] = plotting_df['ROI'].str.extract('(\d+)').astype(int)
        plotting_df.sort_values('ROI_num', inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use a predefined seaborn palette for nicer colors
        palette = sns.color_palette("tab10", n_colors=len(plotting_df['Model'].unique()))

        sns.barplot(data=plotting_df, x="ROI", y=metric, hue="Model", palette=palette, alpha=.6, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        
        # Decorate the plot with error bars, significance markers, and noise ceiling
        self.decorate_plot(ax, plotting_df, metric, pairs)

        ax.set_title("R2 for best performing layer", fontsize=16)
        plt.subplots_adjust(right=0.85)
        plt.show()  # Ensure this is the only plt.show() in this function

        # Optionally return the DataFrame of best layers
        best_layers_df = plotting_df[['ROI', 'Model', 'Layer', metric]].drop_duplicates().reset_index(drop=True)
        return best_layers_df


    def decorate_plot(self, ax, plotting_df, metric, pairs):
        """Decorate the plot with error bars, significance markers, and noise ceiling."""
        for i, patch in enumerate(ax.patches):
            # Calculate the x position and height of the bar
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            
            # Adjust error bars to not go below zero
            if i < len(plotting_df):  # Ensure we're not exceeding the DataFrame's length
                row = plotting_df.iloc[i]
                lower_limit = max(y - row["SEM"], 0)  # Ensure the lower limit is not below 0
                upper_limit = y + row["SEM"]
                ax.errorbar(x, y, yerr=[[y-lower_limit], [upper_limit-y]], fmt='none', c='k', capsize=5)
            
                # Significance markers
                if row['Significance'] < 0.05:
                    ax.text(x, upper_limit, '*', ha='center', va='bottom', c='k')


        bar_width = ax.patches[0].get_width()
        for pair in pairs:
            # Extract ROI and Model from the pair tuples
            roi1, model1 = pair[0]
            roi2, model2 = pair[1]

            # Ensure that the pair is from the same ROI
            if roi1 == roi2:
                # Find the indices for the bars corresponding to the pair
                indices = plotting_df[(plotting_df['ROI'] == roi1) & (plotting_df['Model'].isin([model1, model2]))].index.tolist()

                if len(indices) == 2:  # Check if both bars are found
                    index1, index2 = indices

                    x1 = ax.patches[index1].get_x() + bar_width / 2
                    y1 = ax.patches[index1].get_height()
                    x2 = ax.patches[index2].get_x() + bar_width / 2
                    y2 = ax.patches[index2].get_height()

                    # Calculate the height and tips of the significance line
                    bar_height = max(y1, y2) + 0.05  # Adjusted for visibility
                    bar_tips = bar_height - 0.02  # Adjusted for visibility

                    # Draw the line and asterisk for significance
                    ax.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
                    ax.text((x1 + x2) / 2, bar_height + 0.02, '*', ha='center', va='bottom', c='k')
        

        # Noise ceiling
        self.add_noise_ceiling(ax, plotting_df)


    def plot_all_layers(self, metric='R2', columns_per_row=4):
        rois = pd.concat(self.dataframes)['ROI'].unique()
        n_rois = len(rois)
        rows = int(np.ceil(n_rois / columns_per_row))

        fig, axes = plt.subplots(rows, columns_per_row, figsize=(columns_per_row * 5, rows * 5), squeeze=False)
        axes = axes.flatten()

        bar_width = 0.15  # Define the width for each bar

        for i, roi in enumerate(rois):
            ax = axes[i]
            roi_df = pd.concat([df[df['ROI'] == roi] for df in self.dataframes])

            models = roi_df['Model'].unique()
            layers = roi_df['Layer'].unique()
            n_models = len(models)
            n_layers = len(layers)

            # Select base colors for models and generate darker shades for layers
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Example base colors
            base_colors = base_colors[:n_models]  # Ensure we have enough colors for the models

            for j, model in enumerate(models):
                model_df = roi_df[roi_df['Model'] == model]

                # Generate darker shades for the layers within the model's base color
                layer_colors = sns.dark_palette(base_colors[j], n_colors=n_layers + 2, reverse=True)[1:-1]

                for k, layer in enumerate(layers):
                    layer_df = model_df[model_df['Layer'] == layer]

                    # Calculate x position for each bar within the group
                    x_pos = j * (n_layers * bar_width + bar_width) + k * bar_width

                    if not layer_df.empty:
                        ax.bar(x_pos, layer_df[metric].values, width=bar_width, label=f'{layer}' if i == 0 else "", color=layer_colors[k])
                        ax.errorbar(x_pos, layer_df[metric].values, yerr=layer_df['SEM'].values, fmt='none', ecolor='black', capsize=5, capthick=2)

                        if layer_df['Significance'].values < 0.05:
                            ax.text(x_pos, layer_df[metric].values + layer_df['SEM'].values, '*', ha='center', va='bottom', color='black')

            # Noise ceiling lines for the entire ROI
            lnc, unc = roi_df[['LNC', 'UNC']].iloc[0]
            ax.hlines(y=lnc, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='dotted', alpha=0.7, label='LNC' if i == 0 else "")
            ax.hlines(y=unc, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='dashed', alpha=0.7, label='UNC' if i == 0 else "")

            ax.set_xticks(np.arange(n_models) * (n_layers * bar_width + bar_width))
            ax.set_xticklabels(models)
            ax.set_title(f'All Layers for {roi}')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)

        # Hide unused subplots
        for j in range(i + 1, rows * columns_per_row):
            axes[j].axis('off')

        # Add a legend outside the plot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Layers & Noise Ceiling', loc='upper right', bbox_to_anchor=(1.05, 1), ncol=1)

        plt.tight_layout()
        plt.show()




    def decorate_subplot(self, ax, df, metric):
        for i, patch in enumerate(ax.patches):
            x = patch.get_x() + patch.get_width() / 2
            height = patch.get_height()

            if height > 0:  # Ensure the bar has a positive height
                row = df.iloc[i % len(df)]  # Cycle through the DataFrame rows for each patch
                if row['Significance'] < 0.05:
                    ax.text(x, height, '*', ha='center', va='bottom', color='black')

                sem = row['SEM'] if height - row['SEM'] > 0 else height  # Prevent error bars from going below 0
                ax.errorbar(x, height, yerr=[[sem], [sem]], fmt='none', c='k', capsize=5)

        # Noise ceiling
        for unc, lnc in zip(df['UNC'], df['LNC']):
            ax.hlines(y=unc, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='dashed', alpha=0.7)
            ax.hlines(y=lnc, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='dotted', alpha=0.7)

        ax.set_ylim(bottom=0)  # Ensure the plot does not go below 0
        ax.legend([], [], frameon=False)


    def is_2d_array(self, value):
        """Check if the input value is a 2D numpy array."""
        return isinstance(value, np.ndarray) and len(value.shape) == 2

    def add_std_deviation(self, dataframe):
        """Add standard deviation to the dataframe if the values are 2D arrays."""
        dataframe['Std'] = dataframe["Values"].apply(lambda x: np.std(x, axis=0) if self.is_2d_array(x) else None)
        return dataframe


    def plotting_over_time(self, add_std=False):
        """Plotting line plots over time for each dataframe."""
        for dataframe in self.dataframes:
            self.add_std_deviation(dataframe)

            # Average 2D arrays over the first axis
            dataframe["Values_plotting"] = dataframe["Values"].apply(lambda x: np.mean(x, axis=0) if self.is_2d_array(x) else x)

            # Extract time points from the first entry
            sample_value = dataframe.iloc[0]["Values_plotting"]
            time_points = range(len(sample_value))

            # Define a color palette
            palette = sns.color_palette("husl", n_colors=len(dataframe)+2)

            # Initialize the plot
            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")
            plt.style.use('ggplot')

            # Plot lines for each entry in the dataframe
            for index, row in dataframe.iterrows():
                name = row["Description"]
                values = row["Values_plotting"]
                std = row["Std"]
                color = palette[index]

                # Plot values with optional standard deviation shading
                plt.plot(time_points, values, label=name, color=color, linewidth=2)
                if add_std and std is not None:
                    plt.fill_between(time_points, values - std, values + std, color=color, alpha=0.2)

                # Plot significant time points below the x-axis
                if "Significance" in row:
                    sig_indices = [i for i, sig in enumerate(row["Significance"]) if sig < 0.01]
                    for i in sig_indices:
                        plt.hlines(y=-0.001 * (index + 1), xmin=time_points[i], xmax=time_points[i]+1, colors=color, linewidth=2)

            # Add dashed lines at 0 for x and y axes
            plt.axhline(0, color="black", linestyle="--")
            plt.axvline(0, color="black", linestyle="--")

            # Set labels and title
            plt.xlabel("Time (ms)")
            plt.ylabel("Measure")
            plt.title("Time Series Analysis Results")

            # Add legend and show plot
            plt.legend()
            plt.show()