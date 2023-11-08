=======================================================
Evaluation
=======================================================


After the generating RDMs you can use these to evaluate them against brain data. **Net2Brain** contains 3 datasets to play around with.
In case you have your own dataset, make sure the brain activations are in this format: **FORMAT FEHLT**



Representational Similiratiy Analysis (RSA)
----------------

To perform RSA you can load up the RSA module. It will return a dataframe containing all correlation values, and information about the statistical significance.
RSA requires:

- **model_rdms_path**: Path to model RDMs
- **brain_rdms_path**: Path to brain RDMs
- **model_name**: Name of model (important for plotting)

.. code-block:: python

    # AlexNet Example
    from net2brain.evaluations.rsa import RSA
    
    # Start RSA
    evaluation_alexnet = RSA(model_rdms, brain_rdms, model_name="AlexNet")

    # Evaluation - Returns a pandas dataframe
    dataframe1 = evaluation_alexnet.evaluate() 

    # Show results
    display(dataframe1)


Significant pairs
^^^^^^^^^^^^^^^^^^^^
Furthermore, you might be interested in determining whether one model is significantly better than another, and not merely due to random variation. In this case, you can utilize the `compare_model` functionality provided by the toolbox. Use the following syntax:



.. code-block:: python

    test, sig_pairs = eval_1.compare_model(eval_2)


Plotting RSA Results
----------------
The integrated plotting functionality of the toolbox allows you to easily visualize evaluation results. To achieve this, initialize the class with a list of DataFrames obtained from the evaluation. Make sure that each DataFrame:

1. Contains the same ROIs, signifying that each test was performed on the same brain RDMs.
2. Has a distinct model name, which can be set manually or through the "model_name" parameter during evaluation (as mentioned earlier).

The Plotting class requires:

- **dataframes**: One dataframe from the evaluation, or list of dataframes

The function for plotting takes:

- **pairs**: (Optional) List of significant pairs that can be calculated using the evaluation function
- **metric**: (Optional) Either R2 or R 



.. code-block:: python

    from net2brain.evaluations.plotting import Plotting

    plotter = Plotting([dataframe1])
    results_dataframe = plotter.plot()

    # Or
    plotter = Plotting([dataframe1,dataframe2])
    results_dataframe = plotter.plot(pairs=sig_pairs)




Weighted RSA
----------------
In addition to the standard RSA, Net2Brain also supports weighted RSA (WRSA) as an evaluation metric. 
WRSA allows for the incorporation of weights into the analysis, providing an alternative approach to evaluating model performance and examining the relationship between neural representations 
and computational models.


WRSA requires:

- **model_rdms_path**: Path to model RDMs
- **brain_rdms_path**: Path to brain RDMs
- **model_name**: Name of model (important for plotting)

.. code-block:: python

    # Start WRSA
    evaluation = WRSA(model_rdms, brain_rdms, model_name="ResNet50")

    # Evaluation - Returns a pandas dataframe
    dataframe1 = evaluation.evaluate() 



Searchlight RSA
----------------
The toolbox offers the capability to perform searchlight analysis using Searchlight RDMs in the [ROI, subject, stimuli, stimuli] format. 
Please note that this toolbox does not include RDMs for testing purposes. 
However, if you have access to RDMs, you are welcome to use this functionality to conduct searchlight analysis.

.. code-block:: python

   from net2brain.evaluations.searchlight import Searchlight

   # Start Searchlight
   evaluation = Searchlight(model_rdms, searchlight_rdm)

   # Evaluation - Returns a pandas dataframe
   evaluation.evaluate()



Linear Encoding Models:
----------------
Another integrated analysis pipeline is a linear encoder. Given a npy file with voxel values, and extracted features, 
the encoder performs an X-fold regression where the training data is used to train a PCA embedding and a
linear regression to predict voxel values. The output is the testing split X-fold average pearson correlation.

Linear_encoding requires:

- **feat_path**: Path to model features
- **roi_path**: Path to brain data in RDM format
- **model_name**: Name of model (important for plotting)
- **trn_tst_split**: (Optional) How to split training and testing data at each fold. Defaults to 0.8
- **n_folds** : (Optionsl) Number of folds. Defaults to 3
- **n_components**: (Optional) Number of PCA components. Defaults to 100
- **batch_size**: (Optional) Size of batch of updating the incremental pca. Defaults to 100

.. code-block:: python

    from net2brain.evaluations.encoding import linear_encoding
    
    results_dataframe = linear_encoding(feat_path, roi_path, model_name, n_folds=3, trn_tst_split=0.8, n_components=100, batch_size=100)


Variance Partitioning Analysis (VPA)
----------------

**Net2Brain** also allows Variance Partitioning Analysis as a method for model and brain evaluation. VPA allows **up to 4 independent variables** that can be evaluated
against **one dependent** variable, which would be the brain data.

VPA requires:

- **dependent_variable**: Path to brain data in form of RDM
- **independent_variable**: List of lists, containing paths to RDMs that are part of one group
- **variable_names**: Names of independent variables (important for plotting)


.. code-block:: python

    from net2brain.evaluations.variance_partitioning_analysis import VPA

    independent_variables = [paths_to_RDM_folder_1, paths_to_RDM_folder_2, paths_to_RDM_folder_3, paths_to_RDM_folder_4]
    variable_names = ["Ind_Var1", "Ind_Var2", "Ind_Var3", "Ind_Var4"]

    VPA_eval = VPA(dependent_variable, independent_variables, variable_names)
    dataframe = VPA_eval.evaluate(average_models=True)



Plotting VPA
^^^^^^^^^^^^^^
You can also plot time-data using our plotting functionalty.
If needed the function **plotting_over_time** supports:

- **add_std**: If you wish to see the standard deviation along with the graph. Defaults to False

.. code-block:: python

    from net2brain.evaluations.plotting import Plotting

    # Plotting with significance
    plotter = Plotting(dataframe)

    plotter.plotting_over_time(add_std=True)





