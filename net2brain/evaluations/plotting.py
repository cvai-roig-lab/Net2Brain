import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
plt.style.use('ggplot')


class Plotting:
    """Class for plotting the results generated in the evaulation module
    """
    def __init__(self, dataframes):
        """Initation

        Args:
            dataframes (list): list of pandas dataframes. Need to have the same ROIs
        """
        self.dataframes = dataframes
        pass

    def plot_best_layer(self):
        """This functions plots using the best performing layer

        Returns:
            plotting_df: Pandas dataframe of the data that has been plotted
        """

        # Check if we have the same ROIs for each dataframe:
        check_rois = list(self.dataframes[0]["ROI"].unique())
        for dataframe in self.dataframes:
            these_rois = list(dataframe["ROI"].unique())
            if not check_rois == these_rois:
                print("checked_rois:", check_rois)
                print("ROIs in this dataframe:", these_rois)
                raise ValueError("The dataframes need to have the same ROIs for a comparison")

        # Check if each dataframe has a different model_name:
        models = []
        for dataframe in self.dataframes:
            this_model = dataframe["Model"].unique()
            if models == []:
                models.append(this_model)
            else:
                if this_model not in models:
                    models.append(this_model)
                else:
                    raise ValueError("Each dataframe needs to have a different model name. Change the model names individually or set the model name in the Evaluation-Pipeline to a different name.")

        # Only have the best performing layers per dataframe
        max_dataframes = []
        for dataframe in self.dataframes:
            max_dataframes.append(dataframe.loc[dataframe.groupby('ROI')['R2'].idxmax(), :])

        # Concat dataframes into one dataframe
        plotting_df = pd.concat(max_dataframes, ignore_index=True)

        # Create figure
        fig, ax = plt.subplots(layout='constrained')

        # Iteration dict for plotting grouped barplots
        iter_dict = {}
        significance_dict = {}

        # Collect models and ROIs
        models = list(plotting_df["Model"].unique())
        rois = list(plotting_df["ROI"].unique())

        # Iterate through all the models and collect the specific data
        for model in models:

            # Create sub dataframe
            sub_df = plotting_df[plotting_df["Model"] == model]

            # Add R2 Values
            r2_values = list(sub_df["R2"].values)
            iter_dict[model] = [r2_values]

            # Add Error Values
            error_values = list(sub_df["SEM"].values)
            iter_dict[model].append(error_values)

            # Add Significance Values
            sigs = list(sub_df["Significance"].values)
            significance_dict[model] = sigs

            # Add LNC UNC
            lnc = list(sub_df["LNC"].values)
            unc = list(sub_df["UNC"].values)

        # Plot Grouped Bar-Plots
        x_values = np.arange(len(r2_values))  # the label locations
        width = 0.20  # the width of the bars
        colors = ['#F1404B', '#FF5E57', '#4A90E2', '#6FC3DF', '#F1404B', '#FF5E57', '#4A90E2', '#6FC3DF']

        # Plot Noise Ceiling
        for lnc, unc, x in zip(lnc, unc, x_values):
            ax.fill_between((x - width * 0.5, x + width * (1.5 + (0.5 * len(models)))), lnc, unc, color='gray', label="Noise Ceiling", alpha=0.5)        
        
        for counter,(attribute, measurement) in enumerate(iter_dict.items()):
            offset = width * counter
            rects = ax.bar(x_values + offset, measurement[0], width, yerr=measurement[1], label=attribute, color=colors[counter])

        # Plot Significance Asterix
        for multiplier, (model, significances) in enumerate(significance_dict.items()):
            offset = width * multiplier
            for counter, sig in enumerate(significances):
                if sig < 0.05:
                    ax.text(x_values[counter] + offset, iter_dict[model][0][counter], '*', horizontalalignment='center', verticalalignment='bottom', fontsize=15)

        # Make the plot pretty
        ax.set_title("Results of Evaluation", fontsize=10)
        ax.set_ylabel('$R^{2}$')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_values + width, rois, rotation=45, ha="right")

        # Make sure that labels only appear once in legend (Noise Ceiling)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        return plotting_df

    def plot(self, variant="best_layer"):
        """Depending on the variant it will plot the data

        Args:
            variant (str, optional): Which way to plot. Defaults to "best_layer".

        Returns:
            plotting_df: Pandas dataframe that has been used for plotting
        """

        if variant == "best_layer":
            plotting_df = self.plot_best_layer()
        else:
            raise NotImplementedError(f"Variant {variant} is not available. For now choose 'best_layer'.")
        return plotting_df

    def add_dataframe(self, dataframes):
        """If you want to add more dataframes to the plotting class

        Args:
            dataframes (list): List of pandas dataframes
        """
        for dataframe in dataframes:
            self.dataframes.append(dataframe)
