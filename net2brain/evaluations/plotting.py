import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import seaborn as sns
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
    
    def plot_with_sig(self,pairs):
        # Only have the best performing layers per dataframe
        max_dataframes = []
        for dataframe in self.dataframes:
            max_dataframes.append(dataframe.loc[dataframe.groupby('ROI')['R2'].idxmax(), :])
        plotting_df = pd.concat(max_dataframes, ignore_index=True)
        # Initialize plot
        x = "ROI"
        y = "R2"
        g = sns.catplot(
            data=plotting_df, kind="bar",
            x="ROI", y="R2", hue="Model",
            palette="dark", alpha=.6, height=6
        )
        max_err = max(plotting_df["SEM"])
        # Get coordinates of patches, plot error bars
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.axes.flat[0].patches]
        y_coords = [p.get_height() for p in g.axes.flat[0].patches]
        g.axes.flat[0].errorbar(x=x_coords, y=y_coords, yerr=plotting_df["SEM"], fmt="none", c="#808080")
        bar_width = g.axes.flat[0].patches[0].get_width()
        # Build bar-color to Model name dictionary  
        hue2name_dic = {p.get_fc(): g.legend.get_texts()[ii].get_text() for ii, p in enumerate(g.legend.get_patches())}
        # Get all data points (bar) and sort them
        name_x_y = [(hue2name_dic[p.get_fc()],p.get_x(),p.get_height()) for ii, p in enumerate(g.axes.flat[0].patches)]
        name_x_y_sorted = sorted(name_x_y, key=lambda element: (element[0], element[1]))
        # Get the names of the ROIs being plotted
        ROI_lst = [ROI.get_text() for ROI in g.ax.get_xaxis().get_ticklabels()]
        num_ROIs = len(ROI_lst)
        # Given ROI and Model return xy cooridinates
        pair2xy = {(ROI_lst[ii % num_ROIs],model) : (x,y) for ii,(model,x,y) in enumerate(name_x_y_sorted)}
        # add model comparison significance
        for pair in pairs:
            x1,y1 = pair2xy[pair[0]]
            x2,y2 = pair2xy[pair[1]]
            bar_height = max(y1,y2) + max_err/2 + 0.0005
            bar_tips = bar_height - 0.0002
            plt.plot(
                [x1+0.5*bar_width, x1+0.5*bar_width, x2+0.5*bar_width, x2+0.5*bar_width],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
            )
            plt.text((x1 + x2+bar_width) * 0.5, bar_height+0.001, '*', ha='center', va='bottom', c='k')
        # Get a list of all significant R2 values compared to 0
        sig_list = [(row['ROI'],row['Model']) for index,row in plotting_df.iterrows() if row['Significance'] < 0.05]
        for sig in sig_list:
            x,y = pair2xy[sig]
            plt.text(x + bar_width* 0.5, 0.001, '*', ha='center', va='bottom', c='k')
        num_models = len(g.legend.legend_handles)
        lncl = [plotting_df[plotting_df['ROI']==ROI]['LNC'].iloc[0] for ROI in ROI_lst]
        uncl = [plotting_df[plotting_df['ROI']==ROI]['UNC'].iloc[0] for ROI in ROI_lst]
        label = 'Noise Ceiling'
        for ii,(unc,lnc) in enumerate(zip(uncl,lncl)):
            if unc == lnc:
                continue
            plt.hline(y=unc,xmin=ii-0.5*bar_width*num_models,xmax=ii+0.5*bar_width*num_models,
                linewidth = 1 , color='k', linestyle='dashed' , label = label)
            label = ''
            plt.hline(y=lnc,xmin=ii-0.5*bar_width*num_models,xmax=ii+0.5*bar_width*num_models,
                linewidth = 1 , color='k', linestyle='dashed' , label = label)
        ##############
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        g.legend.remove()
        plt.legend(handles[1:]+[handles[0]],labels[1:]+[labels[0]],loc='center left', bbox_to_anchor=(1, 0.5),title='Models')
        g.despine(left=True)
        g.set_axis_labels("ROI", "R2")
        g.fig.suptitle("Results of Evaluation")
        return plotting_df
    
    def plot(self, variant="best_layer", pairs="None"):
        """Depending on the variant it will plot the data

        Args:
            variant (str, optional): Which way to plot. Defaults to "best_layer".

        Returns:
            plotting_df: Pandas dataframe that has been used for plotting
        """

        if variant == "best_layer":
            plotting_df = self.plot_best_layer()
        elif variant == "significance":
            plotting_df = self.plot_with_sig(pairs=pairs)
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
