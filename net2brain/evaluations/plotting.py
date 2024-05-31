import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import re
import numpy as np
import matplotlib.pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('seaborn-v0_8-whitegrid')


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
        try:
            plotting_df['ROI_num'] = plotting_df['ROI'].str.extract('\((\d+)\)').astype(int)
            plotting_df = plotting_df.sort_values('ROI_num').reset_index(drop=True)
        except ValueError:
            plotting_df = plotting_df.sort_values('ROI').reset_index(drop=True)

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



    def plot_all_layers(self, metric='R2', columns_per_row=4, simplified_legend=False):
        for dataframe in self.dataframes:
            dataframe = self.prepare_dataframe(dataframe, metric)
        rois = pd.concat(self.dataframes)['ROI'].unique()
        n_rois = len(rois)
        rows = int(np.ceil((n_rois) / columns_per_row))

        fig, axes = plt.subplots(rows, columns_per_row, figsize=(columns_per_row * 9, rows * 4), squeeze=False)
        axes = axes.flatten()

        all_handles_labels = []

        for i, roi in enumerate(rois):
            ax = axes[i]
            roi_df = pd.concat([df[df['ROI'] == roi] for df in self.dataframes])

            try:
                roi_df['Layer_num'] = roi_df['Layer'].str.extract('\((\d+)\)').astype(int)
                roi_df = roi_df.sort_values(['Model', 'Layer_num']).reset_index(drop=True)
            except ValueError:
                roi_df = roi_df.sort_values(['Model', 'Layer']).reset_index(drop=True)

            models = roi_df['Model'].unique()
            layers = roi_df['Layer'].unique()
            n_models = len(models)
            n_layers = len(layers)

            bar_width = 0.4 / n_layers  # Adjusted bar width for less space between bars
            model_spacing = 0.1  # Adjusted spacing between models

            # Calculate positions for each bar and the middle position for each model's group
            model_positions = []
            current_pos = 0  # Start from zero and increment
            for j, model in enumerate(models):
                model_df = roi_df[roi_df['Model'] == model]
                model_layers = model_df['Layer'].unique()
                start_pos = current_pos  # Start position for this model group
                end_pos = start_pos + (len(model_layers) - 1) * bar_width
                middle_pos = (start_pos + end_pos) / 2
                model_positions.append(middle_pos)

                layer_colors = sns.dark_palette(sns.color_palette("tab10")[j % len(sns.color_palette("tab10"))], n_colors=len(model_layers) + 2)[1:-1]

                for k, layer in enumerate(model_layers):
                    layer_df = model_df[model_df['Layer'] == layer]
                    x_pos = start_pos + k * bar_width

                    if not layer_df.empty:
                        bar = ax.bar(x_pos, layer_df[metric].values, width=bar_width, label=f'{model} {layer}' if i == 0 else "", color=layer_colors[k])
                        ax.errorbar(x_pos, layer_df[metric].values, yerr=layer_df['SEM'].values, fmt='none', ecolor='black', capsize=5, capthick=2)

                        if layer_df['Significance'].values < 0.05:
                            ax.text(x_pos, layer_df[metric].values + layer_df['SEM'].values, '*', ha='center', va='bottom', color='black')

                    current_pos = end_pos + model_spacing  # Update position for the next model group

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                all_handles_labels.append((handles, labels))

            ax.set_xticks(model_positions)
            ax.set_xticklabels(models, fontsize=14)
            ax.set_title(f'Correlation Analysis for {roi}', fontsize=14)
            ax.set_xlabel('Models with layers', fontsize=14)
            ax.set_ylabel('Correlation Coefficient (R)', fontsize=14)

        for j in range(i + 1, rows * columns_per_row):
            axes[j].axis('off')

        # Collect handles and labels for the legend
        if simplified_legend:
            for ax in axes[:n_rois]:
                # Add a textbox at the upper left position of each axis
                textstr = "Gradient: Early (darker) to Later (brighter) layers"
                props = dict(boxstyle='round', facecolor='white', edgecolor='black')
                ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)



        else:
            handles, labels = zip(*all_handles_labels)
            handles = [h for sublist in handles for h in sublist]
            labels = [l for sublist in labels for l in sublist]

            network_handles = {}
            for handle, label in zip(handles, labels):
                model_name = label.split()[0]
                if model_name not in network_handles:
                    network_handles[model_name] = []
                network_handles[model_name].append((handle, label))

            new_handles = []
            new_labels = []
            for model_name in sorted(network_handles.keys()):
                model_handles_labels = network_handles[model_name]
                try:
                    # Extract digits from labels like 'RDM_features_10.npz'
                    sorted_model_handles_labels = sorted(model_handles_labels, key=lambda x: int(re.search(r'\d+', x[1].split('_')[-1]).group()))
                except AttributeError:
                    # Some layers are not numbered (AttributeError: 'NoneType' object has no attribute 'group')
                    sorted_model_handles_labels = sorted(model_handles_labels, key=lambda x: x[1])
                for handle, label in sorted_model_handles_labels:
                    new_handles.append(handle)
                    new_labels.append(label)

            legend_handles = new_handles
            legend_labels = new_labels
            legend_columns = n_models

            # Add a single legend at the bottom of the figure
            fig.legend(legend_handles, legend_labels, loc='upper center', ncol=legend_columns, fontsize=14, title='Model Layers', title_fontsize=14, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 1])
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