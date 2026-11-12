=======================================================
Evaluation
=======================================================

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/3_Evaluation.ipynb>`_!



Utilizing the generated representational dissimilarity matrices (RDMs), one can conduct a comprehensive evaluation against neural data. **Net2Brain** provides access to three distinct datasets for exploration purposes. For those possessing custom datasets, ensure the neural activation data adheres to the following formats: *[subject, stimuli, stimuli]* or *[subject, time, stimuli, stimuli]*.

Representational Similarity Analysis (RSA)
------------------------------------------

The RSA module within **Net2Brain** simplifies the performance of RSA, offering a DataFrame output with correlation values and statistical significance markers. For RSA execution, supply:

- **model_rdms_path**: Directory containing the model RDMs.
- **brain_rdms_path**: Directory containing the brain RDMs.
- **model_name**: Identifier for the model, significant for plotting.

.. code-block:: python

    # Example using AlexNet
    from net2brain.evaluations.rsa import RSA
    
    # Initialize RSA
    rsa_evaluation = RSA(model_rdms_path, brain_rdms_path, model_name="AlexNet")
    
    # Conduct Evaluation - Yields a DataFrame
    results_df = rsa_evaluation.evaluate()
    
    # Display the Results
    print(results_df)

With ``squared=True`` (default) the DataFrame holds the mean over subjects of r² (``R2``), the per-subject values
(``R2_array``) and ``%R2`` (relative to the lower noise ceiling ``LNC``); with ``squared=False`` the same for r (``R``,
``R_array``, ``%R``). ``Significance`` is the p-value of a one-sided one-sample t-test of the per-subject correlations
r against 0 (mean r > 0), also when ``squared=True``. For MEG/EEG data the correlations and the noise ceiling are
computed per timepoint and averaged over timepoints if ``timepoint_agg=True`` (else one value per timepoint).

``evaluate`` takes two optional arguments:

- **distance_metric**: How the RDMs are compared, ``"spearman"`` (default) or ``"pearson"``. Other values raise a ValueError.
- **correction**: ``None`` (default, uncorrected) or ``"bonferroni"`` (over the layers of each ROI, capped at 1). Other values raise a ValueError.

Significant Model Comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the objective is to ascertain whether one model outperforms another significantly and not by mere chance, the `compare_model` feature comes into play.

.. code-block:: python

    test_statistic, significant_pairs = rsa_evaluation.compare_model(another_evaluation)

``compare_model`` scores both models with the ``distance_metric`` of the last ``evaluate()`` call of
``rsa_evaluation`` (Spearman without one); MEG/EEG data need ``timepoint_agg=True`` and ``model_timepoints=False``. It
chooses the best layer of each model on a random half of the subjects (fixed seed) and compares the two chosen layers
with a paired t-test of the per-subject r on the other half (also when ``squared=True``). Choosing and testing on the
same subjects would favour the model with more layers. The split halves the number of test subjects, so the test has
little power with few subjects. ``test_statistic`` holds ``(tstat, p)`` per ROI (``tstat`` is positive if
``another_evaluation`` is better).

Visualizing RSA Results
-----------------------

The built-in plotting capabilities of **Net2Brain** facilitate the visualization of evaluation metrics. Initialize the `Plotting` class with the evaluation DataFrames, ensuring each one:

1. Shares identical regions of interest (ROIs) indicating tests on the same brain RDMs.
2. Features a unique model identifier, set manually or via the "model_name" during the evaluation process.

Required for plotting:

- **dataframes**: A single DataFrame from the evaluation or a list of them.
- **pairs**: (Optional) A list of significant pairs derived from the evaluation function.
- **metric**: (Optional) ``"R2"``, ``"R"`` or ``"Distance"``. Defaults to the first of these that all DataFrames contain. Values are plotted as stored (there is no conversion between R and R²), so ``metric="R"`` needs an ``R`` column, e.g. RSA with ``squared=False``. For ``"Distance"`` (distributional comparisons) the lowest value is the best layer.

.. code-block:: python

    from net2brain.evaluations.plotting import Plotting
    
    # Single DataFrame plotting
    single_plotter = Plotting([results_df])
    single_plotter.plot()

    # Plotting all layers
    single_plotter.plot_all_layers()
    
    # Multiple DataFrames plotting
    multiple_plotter = Plotting([results_df, another_results_df])
    multiple_plotter.plot(pairs=significant_pairs)




Weighted RSA
----------------

**Net2Brain** extends support for Weighted RSA, which combines the RDMs of all layers of a model into one weighted
model RDM with non-negative weights. The stimuli are split once at random (fixed seed) into ``min(5, n_stimuli // 3)``
folds. For every fold and every held-out subject, the weights are fit (least squares with intercept) on the mean RDM of
the other subjects, using only the pairs of the training stimuli, and the weighted RDM is correlated (Pearson) with the
held-out subject's RDM on the pairs of the held-out stimuli. ``R2`` is the mean over subjects of r² (r averaged over the
folds), ``Significance`` a one-sided t-test of r against 0 over subjects, and ``LNC``/``UNC`` are Pearson noise ceilings
on the same held-out pairs, aggregated like ``R2``. The output has one row per ROI (``Layer`` "All"). MEG/EEG RDMs are averaged over time
first. ``R2`` and ``Significance`` depend on the random split of the stimuli; for weak effects they can differ
considerably between splits. At least 6 stimuli and 2 subjects are needed; ``squared``, ``distance_metric`` and
``ridge_weight`` are not used.

WRSA necessitates:

- **model_rdms_path**: Directory containing the model RDMs.
- **brain_rdms_path**: Directory containing the brain RDMs.
- **model_name**: Identifier for the model, significant for plotting.

.. code-block:: python

    # Example with WRSA
    from net2brain.evaluations.weighted_rsa import WRSA
    
    # Initialize WRSA
    wrsa_evaluation = WRSA(model_rdms_path, brain_rdms_path, model_name="ResNet50")
    
    # Perform Evaluation - Produces a DataFrame
    wrsa_results_df = wrsa_evaluation.evaluate()
    
    # Output the Results
    print(wrsa_results_df)



Searchlight RSA
---------------

For a focused analysis, the toolbox offers a searchlight RSA functionality, which requires Searchlight 
RDMs formatted as *[ROI, subject, stimuli, stimuli]*. Note that this toolbox does not provide RDMs for demonstration purposes,
but users can apply the feature if they have suitable RDMs.

.. code-block:: python

    from net2brain.evaluations.searchlight import Searchlight
    
    # Initialize Searchlight (distance_metric: "spearman", the default, or "pearson")
    searchlight_evaluation = Searchlight(model_rdms_path, searchlight_rdms_path)
    
    # Conduct Evaluation - Returns a dictionary
    searchlight_results = searchlight_evaluation.evaluate()
    
    # Present the Results
    print(searchlight_results)



Linear Encoding Models:
----------------

Another integrated analysis pipeline is a linear encoder. Given a npy file with voxel values, and extracted features, the encoder performs an X-fold regression where the training data is used to train a PCA embedding and a linear regression to predict voxel values. The output is the testing split X-fold average pearson correlation.

The feature files are paired with the rows of the brain data in natural order of their file names (``stimulus_2``
before ``stimulus_10``), as in RDM creation; a warning is shown if this differs from alphabetical order, which earlier
versions used. A different number of feature files and brain-data rows raises an error.
``Significance`` is NaN for ``Linear_Encoding``, ``Ridge_Encoding`` and ``Stacked_Encoding``: the earlier t-tests across
folds, voxels or ROIs treated dependent values as independent, and no valid test is implemented.

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/Workshops/Net2Brain_Introduction_LLM.ipynb>`_!


Prerequisites for the linear encoding function include:

- **feat_path**: The file path directing to the model's features.
- **roi_path**: The file path for brain data (``.npy``, stimuli x voxels).
- **model_name**: The identifier for the model, crucial for subsequent visualization.
- **trn_tst_split**: (Optional) specifies the train-test data ratio per fold, with a default of 0.8.
- **n_folds**: (Optional) The cross-validation folds count, preset to 3.
- **n_components**: (Optional) The PCA components quantity, defaulting to 100.
- **batch_size**: (Optional) The incremental PCA's batch size, with a standard value of 100.
- **pooling** (Optional): Pooling method for variable-length features. Options: ``'mean'`` (average across sequence), ``'max'`` (maximum values across sequence), ``'first'`` (use first token/position, e.g., [CLS] token), ``'last'`` (use final token/position). Required when processing transformer/LLM features with different sequence lengths. Defaults to ``None`` for fixed-length features.


.. code-block:: python

    from net2brain.evaluations.encoding import Linear_Encoding
    
    results_dataframe = Linear_Encoding(
        feat_path,
        roi_path,
        model_name,
        n_folds=3,
        trn_tst_split=0.8,
        n_components=100,
        batch_size=100,
        return_correlations=True,
        save_path="path/to/csv"
    )




Variance Partitioning Analysis (VPA)
------------------------------------

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/Workshops/Net2Brain_EEG_Cutting_Edge_Workshop.ipynb>`_!



**Net2Brain** enhances model and cerebral data assessment through Variance Partitioning Analysis.
This technique supports the evaluation of **up to four independent variables** in relation to a
**singular dependent variable**, typically the neural data. It works both for time-resolved data
(e.g. EEG/MEG) and for data without a time axis (e.g. fMRI).

The requirements for VPA are:

- **dependent_variable**: The brain data. For time-resolved data (e.g. EEG) this is a single path to a pre-stacked array shaped *[subjects, time, pairs]*. For data without a time axis (e.g. fMRI) it can be either a single *[subjects, pairs]* array or a list of per-subject RDM paths; in that case the results are returned per subject, with no time dimension.
- **independent_variable**: A list with one entry per independent variable. Each entry is itself a list of RDM paths belonging to that group, which are averaged together when ``average_models=True``. A bare path string is also accepted for a single-model group.
- **variable_names**: The labels for the independent variables, integral for visualization.

Optional arguments of ``evaluate``:

- **average_models**: Average the RDMs within each independent variable. Defaults to False.
- **n_permutations**: Number of permutations for the significance test. Defaults to 1000; 0 skips the test (Significance is NaN).
- **seed**: Seed of the permutations. Defaults to 0.

Returns:

- **dataframe**: Contains the R² of every combination of independent variables (``R1``, ``R12``, ...), all unique and shared variances (``y1``, ``y12``, ...), plus the upper and lower noise ceiling as ``UNC`` and ``LNC`` rows. The dataframe can be filtered to only contain relevant information.

Significance is computed with a permutation test: the stimulus labels of the independent variables are permuted
(the same permutation for all of them) and the mean over subjects is compared with the permutation null. For
time-resolved data the null is the maximum over timepoints, so the p-values are corrected for all timepoints.
p-values are given for every R² row (``R1``, ``R12``, ..., each tested on its own: they are corrected over
timepoints, not across rows, so correct for the number of rows you test yourself). The unique and shared variances
(``y1``, ``y12``, ...) have NaN as significance: the permutation tests whether any variable relates to the brain, not
whether one variable adds to the others, so it would give false positives for them. The test generalizes over stimuli;
the subjects are treated as fixed. If R² is NaN for a brain RDM (NaN or infinite values, or an exactly constant
RDM), the p-values are NaN and a warning is shown.



.. code-block:: python

    from net2brain.evaluations.variance_partitioning_analysis import VPA

    independent_variables = [paths_to_RDM_folder_1, paths_to_RDM_folder_2, paths_to_RDM_folder_3, paths_to_RDM_folder_4]
    variable_names = ["Ind_Var1", "Ind_Var2", "Ind_Var3", "Ind_Var4"]

    VPA_eval = VPA(dependent_variable, independent_variables, variable_names)
    dataframe = VPA_eval.evaluate(average_models=True)

    # Keep the R² of the full model (it has p-values) and the unique variances (no p-values)
    dataframe = dataframe.query("Variable in ['R1234', 'y1', 'y2', 'y3', 'y4']").reset_index(drop=True)




Plotting VPA
^^^^^^^^^^^^^^
The plotting utilities of **Net2Brain** cover both kinds of VPA results, and both overlay the
upper and lower noise ceiling.

For time-resolved data, ``plotting_over_time`` draws each component as a line over time and
supports an optional standard-deviation band.

- **add_std**: Enable to display the standard deviation on the graph. Defaults to False.

.. code-block:: python

    from net2brain.evaluations.plotting import Plotting

    # Time-resolved results (e.g. EEG)
    plotter = Plotting(dataframe)
    plotter.plotting_over_time(add_std=True)

For data without a time axis, ``plotting_components`` draws each component as a bar (mean across
subjects with SEM error bars) and marks significance.

- **threshold**: Significance level for the markers. Defaults to 0.05.

.. code-block:: python

    # Non-time-resolved results (e.g. fMRI)
    plotter = Plotting(dataframe)
    plotter.plotting_components()



Centered Kernel Alignment (CKA)
----------------

Centered Kernel Alignment (CKA) is a similarity metric used to compare two datasets, such as DNN-derived features and brain activity, based on the relationships within each dataset. Unlike traditional correlations, CKA is scale-invariant and focuses on the structure of pairwise similarities within each dataset.

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/3_Evaluation.ipynb>`_!

Prerequisites for the CKA function include:

- **feat_path**: The file path directing to the model's feature `.npz` files, where each file contains multiple layer activations.
- **brain_path**: The file path for `.npy` files containing brain data, with each file representing a specific ROI.
- **model_name**: The identifier for the model, crucial for labeling the output.

Returns:
- **dataframe**: Contains CKA scores for each ROI and each layer of the DNN.

.. code-block:: python

    from net2brain.evaluations.cka import CKA

    results_dataframe = CKA.run(
        feat_path="path/to/features",
        brain_path="path/to/brain_data",
        model_name="model_name"
    )


Distributional Comparison (DC)
----------------

Distributional Comparison evaluates the similarity between datasets by comparing the feature-wise distributions. Two metrics are available:
- **Jensen-Shannon Divergence (JSD):** Measures the divergence between two probability distributions. It is symmetric and bounded between 0 and 1.
- **Wasserstein Distance (WD):** Also known as Earth Mover's Distance, measures the cost of transforming one distribution into the other.

Each feature is standardized; JSD (base 2) is computed on histograms with shared bins, Wasserstein on the samples. The distances are averaged across features.

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/3_Evaluation.ipynb>`_!

Prerequisites for the Distributional Comparison function include:

- **feat_path**: The file path directing to the model's feature `.npz` files, where each file contains multiple layer activations.
- **brain_path**: The file path for `.npy` files containing brain data, with each file representing a specific ROI.
- **metric**: The distance metric used to compare distributions. Options are:
  - **"jsd"**: Jensen-Shannon Divergence.
  - **"wasserstein"**: Wasserstein Distance.
- **"bins":**  Number of bins for histogramming (JSD only), defaults to 50.
- **model_name**: The identifier for the model, crucial for labeling the output.

.. warning::

   Both the feature data and the brain data are reduced via PCA to ``min(n_features, n_voxels, n_stimuli - 1)`` components. While this ensures compatibility, it alters the feature space.

Returns:
- **dataframe**: Contains distributional comparison scores in the ``Distance`` column (lower means more similar) for each ROI and each layer of the DNN; ``Significance``, ``SEM``, ``LNC`` and ``UNC`` are NaN.

.. code-block:: python

    from net2brain.evaluations.distributional_comparisons import DistributionalComparison

    # Using Jensen-Shannon Divergence (JSD)
    results_jsd = DistributionalComparison.run(
        feat_path="path/to/features",
        brain_path="path/to/brain_data",
        metric="jsd",
        bins=50,
        model_name="model_name"
    )

    # Using Wasserstein Distance (WD)
    results_wasserstein = DistributionalComparison.run(
        feat_path="path/to/features",
        brain_path="path/to/brain_data",
        metric="wasserstein",
        bins=50,
        model_name="model_name"
    )

Stacked Encoding
---------------------

Stacked Encoding combines predictions from multiple feature spaces (such as different neural network layers) to improve brain activity prediction. Instead of using a single layer or concatenating features, stacked encoding:

- Trains separate ridge regression models for each feature space.
- Learns optimal weights to combine these predictions.
- Produces weights that indicate each feature space's importance.

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/3_Evaluation.ipynb>`_!

Prerequisites for the Stacked Encoding function include:

- **feat_path**: Path to model activation files (`.npz`), with one file per layer.
- **roi_path**: Path to fMRI data files (`.npy`), or a list of paths for multiple ROIs/subjects.
- **model_name**: Identifier for the model (e.g., 'AlexNet', 'ResNet50') for labeling results.
- **n_folds**: Number of cross-validation folds for stacking (higher = more robust, but slower).
- **n_components**: Number of principal components for dimensionality reduction (`None` to skip).
- **vpa**: Whether to perform variance partitioning analysis automatically.
- **save_path**: Directory to save encoding results and variance partitioning data.

Returns:
- **dataframe**: Contains model performance metrics per layer and ROI. ``R`` is the Pearson r between the out-of-fold predictions and the data, averaged over voxels; ``Significance`` is NaN.

.. code-block:: python

    from net2brain.evaluations.stacked_encoding import Stacked_Encoding

    # Basic usage
    results_df = Stacked_Encoding(
        feat_path="path/to/model/features/",
        roi_path="path/to/brain/data/",
        model_name="AlexNet",
        n_folds=5,
        n_components=100,
        vpa=True,
        save_path="results/alexnet_encoding"
    )

    # Inspect results
    print(results_df.head())

---

Stacked Variance Partitioning
-----------------------------------

Stacked Variance Partitioning (structured variance partitioning, Lin et al.) analyzes which layers of a network are needed to predict
brain activity. It compares the full stacked model with stacked models on nested sets of layers, ordered from the first to the last layer:

- **Forward direction**: Layers are removed from the first one on (models on layers k+1..K). A voxel is assigned the first layer whose removal (together with the layers before it) drops the R² below 95% of the full model, i.e. the lowest layer that is still needed.

- **Backward direction**: Layers are removed from the last one on (models on layers 1..K-k), giving the highest layer that is still needed.

Together they define an *interval* of relevant layers for each voxel. The assignment is not a significance test: every voxel where the full
model has R² > 0 gets a layer (the others get NaN), so restrict the maps to voxels where the full model predicts well.

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/3_Evaluation.ipynb>`_!

Prerequisites for the Stacked Variance Partitioning function include:

- **nested_r2s_forward**: R² of the forward nested stacked models (`n_layers × n_voxels`), row k = layers k+1..K.
- **nested_r2s_backward**: R² of the backward nested stacked models (`n_layers × n_voxels`), row k = layers 1..K-k.
- **save_path**: Path to save variance partitioning results as an `.npz` file (optional; by default nothing is saved).

Both are computed by ``nested_stacked_r2s(data, features)`` and saved by ``Stacked_Encoding(vpa=True)`` as ``r2s_array`` and ``r2sr``.


Returns:
- **dictionary**: Contains variance partitioning results, including layer assignments.

.. code-block:: python

    from net2brain.evaluations.stacked_encoding import nested_stacked_r2s
    from net2brain.evaluations.stacked_variance_partitioning import Stacked_Variance_Partitioning

    # brain_data: n_stimuli x n_voxels, features: list of n_stimuli x n_dim arrays from the first to the last layer
    forward, backward = nested_stacked_r2s(brain_data, [layer1_features, layer2_features, layer3_features], n_folds=5)

    # Perform variance partitioning
    vp_results = Stacked_Variance_Partitioning(forward, backward, save_path="results/variance_partitioning")

    # Access the results
    forward_layers = vp_results["vp_sel_layer_forward"]   # Layers selected in forward analysis
    backward_layers = vp_results["vpr_sel_layer_backward"]  # Layers selected in backward analysis


