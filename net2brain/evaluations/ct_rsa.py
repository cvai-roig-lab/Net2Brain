import numpy as np
import pandas as pd
from .rsa import RSA


class CTRSA():
    def __init__(self, model_rdms_path, brain_rdms_path, model_name, layer_skips=(), squared=False):
        """
        Cross-Temporal RSA (CT-RSA) (Sartzetaki, Zonneveld et al., 2026).
        Evaluation comparing model RDMs to brain RDMs across all model timepoints and layers,
        and identifying the maximum RSA score for each brain timepoint across all model timepoints and layers.
        Brain RDMs should be an array of shape (n_subjects, n_timepoints, n_stimuli, n_stimuli),
        and model RDMs should be computed with the option multi_timepoint_rdms set in RDMCreator.

        Sartzetaki, C., Zonneveld, A.W., Oyarzo, P., Gifford, A.T., Cichy, R.M., Mettes, P. and Groen, I.I.A., 2026.
        The Human Brain as a Dynamic Mixture of Expert Models in Video Understanding. In The Fourteenth
        International Conference on Learning Representations. https://openreview.net/forum?id=bSsNSfyj8m

        Args:
            model_rdms_path (str): Path to the folder containing the model RDMs.
            brain_rdms_path (str): Path to the folder containing the brain RDMs.
            model_name (str): Name of the model.
            layer_skips (tuple, optional): Names of the model layers to skip. Use '_' instead of '.' in the names.
            squared (bool): Whether to square the correlation values.
        """

        self.rsa = RSA(
            model_rdms_path=model_rdms_path,
            brain_rdms_path=brain_rdms_path,
            model_name=model_name,
            layer_skips=layer_skips,
            squared=squared,
            timepoint_agg=False,
            model_timepoints=True,
        )

    def evaluate(self, layer_order=None, baseline_correction=True, sfreq=50, before_onset=0.2):
        """
        Args:
            layer_order: Provide the layer names in the correct depth order if this isn't the case.
            baseline_correction: Whether to do baseline correction on the max scores by subtracting the mean of the baseline window from all scores.
            sfreq: Sampling frequency of the sequential brain data, used to convert timepoint indices to seconds.
            before_onset: Time in seconds before stimulus onset, used to convert timepoint indices to seconds and to define the baseline window for baseline correction.

        Returns:
            final_df: A DataFrame with one row per brain timepoint, containing the maximum RSA score across all model timepoints and layers for that brain timepoint,
            as well as the corresponding layer and model timepoint information. The DataFrame is baseline-corrected if baseline_correction is True.
        """

        df = self.rsa.evaluate()
        df.drop(columns=["Significance"], inplace=True)

        df["Layer"] = (
            df["Layer"]
            .apply(lambda lay: lay.split("RDM_")[1] if "RDM_" in lay else lay)
            .apply(lambda lay: lay.split(".npz")[0] if ".npz" in lay else lay)
        )
        df["Layer_FullNames"] = df["Layer"]
        if not layer_order:
            layer_order = df["Layer"].unique().tolist()
        if self.rsa.layer_skips:
            layer_order = [lay for lay in layer_order if lay not in self.rsa.layer_skips]
        df["Layer"] = df["Layer"].map({lay: i for i, lay in enumerate(layer_order)})
        df.sort_values(by=["Layer"], inplace=True)

        df["R"] = df["R"].apply(np.array)
        max_len = df["R"].apply(len).max()

        def repeat_elements_to_max_length(x):
            repeat_count = max_len // len(x)
            return [item for item in x for _ in range(repeat_count)]

        # Get original n_timepoints for each layer
        all_og_n_time_points = []
        for layer in df["Layer"].unique():
            df_layer = df[df["Layer"] == layer]
            og_n_time_points = len(df_layer['R'].values[0])
            all_og_n_time_points.append(og_n_time_points)

        # make all layers have max timepoints by repeating each timepoint max_timepoints // layer_timepoints times
        df["R"] = df["R"].apply(repeat_elements_to_max_length)

        # Step 1: max layer for all model timepoints and brain timepoints
        max_R_all_model_timepoints = []
        max_layers_all_model_timepoints = []
        max_layers_all_model_timepoints_idx = []
        max_layer_all_model_timepoints_names = []
        for model_timepoint in range(max_len):
            R_matrix = np.stack(df["R"].apply(lambda x: x[model_timepoint]).values)
            max_R = np.max(R_matrix, axis=0)
            max_indices = np.argmax(R_matrix, axis=0)
            max_layer_names = [df.iloc[idx]["Layer_FullNames"] for idx in max_indices]
            layer_values = df["Layer"].values / df["Layer"].max()
            max_layers = [layer_values[idx] for idx in max_indices]
            max_R_all_model_timepoints.append(max_R)
            max_layers_all_model_timepoints.append(max_layers)
            max_layers_all_model_timepoints_idx.append(max_indices)
            max_layer_all_model_timepoints_names.append(max_layer_names)
        row = {
            "Time": np.arange(len(max_R)) / sfreq - before_onset,
            "R": max_R_all_model_timepoints,
            "Layer_at_max_R": max_layers_all_model_timepoints,
            "Layer_at_max_R_idx": max_layers_all_model_timepoints_idx,
            "Layer_at_max_R_names": max_layer_all_model_timepoints_names,
        }
        metadata = df.drop(columns=["Layer", "Layer_FullNames", "R", "%R", "R_array"]).iloc[0].to_dict()

        # Step 2: flatten list-of-lists into long format
        rows = []
        for rep_idx, (R_values_rep, Layer_values_rep) in enumerate(zip(row["R"], row["Layer_at_max_R"])):
            # inner loop over timepoints
            t_idx = 0
            for t, (R_val, Layer_val) in enumerate(zip(R_values_rep, Layer_values_rep)):
                new_row = {
                    "rep_idx_abs": rep_idx,
                    "Time": row["Time"][t],  # row["Time"] is the same for all reps
                    "R": R_val,
                    "Layer_at_max_R": Layer_val,
                    "Layer_at_max_R_idx": row["Layer_at_max_R_idx"][rep_idx][t_idx],
                    "Layer_at_max_R_names": row["Layer_at_max_R_names"][rep_idx][t_idx]
                }
                for k, v in metadata.items():
                    new_row[k] = v
                rows.append(new_row)
                t_idx = t_idx + 1
        long_df = pd.DataFrame(rows)

        # Step 3: max model timepoint for each brain timepoint
        max_dfs = []
        for t in long_df["Time"].unique():
            time_df = long_df[long_df["Time"] == t]
            R_matrix = np.stack(time_df["R"].values)
            max_index = np.argmax(R_matrix, axis=0)
            max_layer = time_df.iloc[max_index]["Layer_at_max_R_idx"]
            max_index_rescaled = int(max_index // (R_matrix.shape[0] / all_og_n_time_points[max_layer]))
            max_slice = time_df.iloc[max_index].copy()
            max_slice['rep_idx_abs'] = max_index_rescaled
            try:  # ratio of best timepoint to total number of timepoints in the original layer (before repeating to max_len)
                max_slice['rep_idx_ratio'] = max_index_rescaled / (all_og_n_time_points[max_layer] - 1)
            except ZeroDivisionError:
                max_slice['rep_idx_ratio'] = 0  # more correct 0.5 to represent the averaging?
            max_dfs.append(max_slice)
        final_df = pd.concat(max_dfs, axis=1).T.reset_index(drop=True)

        # Step 4: do baseline correction on max scores
        if baseline_correction:
            r_values = final_df["R"].values
            times = final_df["Time"].values
            baseline_mask = (times >= -before_onset) & (times <= 0)
            if not np.any(baseline_mask):
                raise ValueError("No timepoints found in the specified baseline window.")
            baseline_mean = np.mean(r_values[baseline_mask])
            corrected_r_values = np.array([r - baseline_mean for r in r_values])
            final_df["R"] = corrected_r_values

        return final_df
