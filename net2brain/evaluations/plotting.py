import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')


class Plotting:
    """Class for plotting the results generated in the evaluation module."""
    
    def __init__(self, dataframes):
        """Initialization

        Args:
            dataframes (list or DataFrame): List of pandas dataframes or a single dataframe. 
                                            If single, it needs to have columns 'Name', 'Values', 'Significance', and 'Color'.
                                            If list, all dataframes need to have the same ROIs.
        """
        
        # Check if dataframes is a single dataframe with the required columns
        if isinstance(dataframes, pd.DataFrame):
            required_columns = ['Name', 'Values', 'Significance', 'Color']
            if all(col in dataframes.columns for col in required_columns):
                self.dataframes = dataframes
                return
            else:
                raise ValueError("The dataframe provided doesn't have all the required columns: 'Name', 'Values', 'Significance', 'Color'.")
        
        # Check if dataframes is a list of dataframes
        elif isinstance(dataframes, list):
            self.dataframes = []
            for dataframe in dataframes:
                if not isinstance(dataframe, pd.DataFrame):
                    raise TypeError(f"Expected all elements in the 'dataframes' list to be of type DataFrame. Found {type(dataframe)} instead.")
                self.dataframes.append(dataframe.copy())
        
        else:
            raise TypeError("The 'dataframes' argument must be either a list of pandas dataframes or a single pandas dataframe with the required columns.")


    def plot(self,pairs=[],metric='R2'):
        for dataframe in self.dataframes:
            if metric not in dataframe.columns:
                if metric == 'R2':
                    dataframe['R2'] = np.power(dataframe['R'],2)
                    dataframe['LNC'] = np.power(dataframe['LNC'],2)
                    dataframe['UNC'] = np.power(dataframe['UNC'],2)
                    dataframe.drop(columns=['R'])
                if metric == 'R':
                    dataframe['R'] = np.power(dataframe['R2'],1/2)
                    dataframe['LNC'] = np.power(dataframe['LNC'],1/2)
                    dataframe['UNC'] = np.power(dataframe['UNC'],1/2)
                    dataframe.drop(columns=['R2'])
        # Only have the best performing layers per dataframe
        max_dataframes = []
        for dataframe in self.dataframes:
            max_dataframes.append(dataframe.loc[dataframe.groupby('ROI')[metric].idxmax(), :])
        plotting_df = pd.concat(max_dataframes, ignore_index=True)
        # Initialize plot
        x = "ROI"
        y = metric
        g = sns.catplot(
            data=plotting_df, kind="bar",
            x="ROI", y=metric, hue="Model",
            palette="dark", alpha=.6, height=6
        )
        g.set_xticklabels(rotation=30) 
        max_err = 0
        # Get coordinates of patches, plot error bars
        if not plotting_df["SEM"].isnull().values.any():
            max_err = max(plotting_df["SEM"])
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
        
        #num_models = len(g.legend.legend_handles)
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        num_models = len(handles)

        lncl = [plotting_df[plotting_df['ROI']==ROI]['LNC'].iloc[0] for ROI in ROI_lst]
        uncl = [plotting_df[plotting_df['ROI']==ROI]['UNC'].iloc[0] for ROI in ROI_lst]
        label = 'Noise Ceiling'
        for ii,(unc,lnc) in enumerate(zip(uncl,lncl)):
            if (unc == lnc) or np.isnan(lnc).any():
                continue

            plt.hlines(y=unc,xmin=ii-0.5*bar_width*num_models,xmax=ii+0.5*bar_width*num_models,
                linewidth = 1 , color='k', linestyle='dashed' , label = label)
            label = ''
            plt.hlines(y=lnc,xmin=ii-0.5*bar_width*num_models,xmax=ii+0.5*bar_width*num_models,

                linewidth = 1 , color='k', linestyle='dashed' , label = label)
        ##############
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        g.legend.remove()
        plt.legend(handles[1:]+[handles[0]],labels[1:]+[labels[0]],loc='center left', bbox_to_anchor=(1, 0.5),title='Models')
        g.despine(left=True)

        g.set_axis_labels("ROI", metric)
        g.fig.suptitle("Results of Evaluation",y=1.01)
        plt.show()
      
        return plotting_df
    

    def is_2d_array(self, value):
        """Function to figure out if the plotting data is 2D or not

        Args:
            value (array or list): Plotting Data

        Returns:
            Bool: True or false
        """
        return isinstance(value, np.ndarray) and len(value.shape) == 2

    def plotting_over_time(self):
        """Plotting lineplots over time
        """

        dataframe = self.dataframes

        # If the values in the "Values" column are 2D, average along axis 0
        dataframe["Values"] = dataframe["Values"].apply(lambda x: np.mean(x, axis=0) if self.is_2d_array(x) else x)

        # Extract time points
        sample_value = dataframe.iloc[0]["Values"]
        time_points = range(len(sample_value))

        # Define a color palette
        palette = sns.color_palette("husl", n_colors=len(dataframe))

        # Initialize the plot
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        plt.style.use('ggplot')

        # Plot lines for each model
        for index, row in dataframe.iterrows():
            name = row["Name"]
            model_values = row["Values"]
            model_significance = row["Significance"]
            color = row.get("Color") or palette[index]

            # Plot values
            plt.plot(time_points, model_values, label=name, color=color, linewidth=2)

            # Calculate the y-coordinate for significant markers; reducing spacing
            y_line_position = -0.001 * (index + 1)

            # Plot significant time points with connected line segments below the x-axis
            sig_indices = [t for t, sig in enumerate(model_significance) if sig < 0.01]
            for i in sig_indices:
                plt.hlines(y=y_line_position, xmin=time_points[i], xmax=time_points[i]+1, colors=color, linewidth=2)

        # Add dashed lines for x and y axes at 0
        plt.axhline(0, color="black", linestyle="--")
        plt.axvline(0, color="black", linestyle="--")

        # Set axis labels and title
        plt.xlabel("Time (ms)")
        plt.ylabel("Unique Variances")
        plt.title("Variance Partitioning Analysis Results")

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()


