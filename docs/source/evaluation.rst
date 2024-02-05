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

Significant Model Comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the objective is to ascertain whether one model outperforms another significantly and not by mere chance, the `compare_model` feature comes into play.

.. code-block:: python

    test_statistic, significant_pairs = rsa_evaluation.compare_model(another_evaluation)

Visualizing RSA Results
-----------------------

The built-in plotting capabilities of **Net2Brain** facilitate the visualization of evaluation metrics. Initialize the `Plotting` class with the evaluation DataFrames, ensuring each one:

1. Shares identical regions of interest (ROIs) indicating tests on the same brain RDMs.
2. Features a unique model identifier, set manually or via the "model_name" during the evaluation process.

Required for plotting:

- **dataframes**: A single DataFrame from the evaluation or a list of them.
- **pairs**: (Optional) A list of significant pairs derived from the evaluation function.
- **metric**: (Optional) The evaluation metric, either R² or R.

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

**Net2Brain** extends support for Weighted RSA, incorporating weights into the analysis to refine model 
performance assessment and the study of neural representation correlations with computational models.

WRSA necessitates:

- **model_rdms_path**: Directory containing the model RDMs.
- **brain_rdms_path**: Directory containing the brain RDMs.
- **model_name**: Identifier for the model, significant for plotting.

.. code-block:: python

    # Example with WRSA
    from net2brain.evaluations.wrsa import WRSA
    
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
    
    # Initialize Searchlight
    searchlight_evaluation = Searchlight(model_rdms_path, searchlight_rdms_path)
    
    # Conduct Evaluation - Returns a DataFrame
    searchlight_results_df = searchlight_evaluation.evaluate()
    
    # Present the Results
    print(searchlight_results_df)



Linear Encoding Models:
----------------

Another integrated analysis pipeline is a linear encoder. Given a npy file with voxel values, and extracted features, the encoder performs an X-fold regression where the training data is used to train a PCA embedding and a linear regression to predict voxel values. The output is the testing split X-fold average pearson correlation.

Prerequisites for the linear encoding function include:

- **feat_path**: The file path directing to the model's features.
- **roi_path**: The file path for brain data formatted as RDM.
- **model_name**: The identifier for the model, crucial for subsequent visualization.
- **trn_tst_split**: (Optional) specifies the train-test data ratio per fold, with a default of 0.8.
- **n_folds**: (Optional) The cross-validation folds count, preset to 3.
- **n_components**: (Optional) The PCA components quantity, defaulting to 100.
- **batch_size**: (Optional) The incremental PCA's batch size, with a standard value of 100.

.. code-block:: python

    from net2brain.evaluations.encoding import linear_encoding
    
    results_dataframe = linear_encoding(
        feat_path,
        roi_path,
        model_name,
        n_folds=3,
        trn_tst_split=0.8,
        n_components=100,
        batch_size=100
    )





Variance Partitioning Analysis (VPA)
----------------

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/Workshops/Net2Brain_EEG_Cutting_Edge_Workshop.ipynb>`_!



**Net2Brain** enhances model and cerebral data assessment through Variance Partitioning Analysis. 
This technique supports the evaluation of **up to four independent variables** in relation to a 
**singular dependent variable**, typically the neural data.

The requirements for VPA are:

- **dependent_variable**: The RDM-formatted path to the brain data.
- **independent_variable**: An array of arrays, each containing RDM paths belonging to a specific group.
- **variable_names**: The labels for the independent variables, integral for visualization.

Returns:
- **dataframe**: Contains all unique and shared variances. Dataframe can be filtered to only contain relevant information



.. code-block:: python

    from net2brain.evaluations.variance_partitioning_analysis import VPA

    independent_variables = [paths_to_RDM_folder_1, paths_to_RDM_folder_2, paths_to_RDM_folder_3, paths_to_RDM_folder_4]
    variable_names = ["Ind_Var1", "Ind_Var2", "Ind_Var3", "Ind_Var4"]

    VPA_eval = VPA(dependent_variable, independent_variables, variable_names)
    dataframe = VPA_eval.evaluate(average_models=True)

    # Filter the dataframe to include only the unique variances and the shared variance by all variables
    dataframe = dataframe.query("Variable in ['y1234', 'y1', 'y2', 'y3', 'y4']").reset_index(drop=True)




Plotting VPA
^^^^^^^^^^^^^^
The plotting utilities of **Net2Brain** offer the capability to visualize time-course data. 
The `plotting_over_time` function includes an optional standard deviation overlay to enrich the
graphical representation.

- **add_std**: Enable to display the standard deviation on the graph. Defaults to False.


.. code-block:: python

    from net2brain.evaluations.plotting import Plotting

    # Plotting with significance
    plotter = Plotting(dataframe)

    plotter.plotting_over_time(add_std=True)





