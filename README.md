# TorchBM 

A pytorch implementation of Boltzmann Machine Learning for the inference of Potts Categorical Models (``n_c > 2 ``) to analyse multidimensional biological data, such as DNA or Protein sequences. 
The implementation includes L1 and L2 regularization. 
***
### Usage

A ``demo_notebook.ipynb `` is provided for example usage on Lattice Proteins (LPs). LPs are artificial protein sequences, used to benchmark the algorithm against ground truth knowledge of their structure. 
Here the dataset if a MSA of $15000$ sequences having lenght ``N_states  = 54 ``, generated using the ground thruth model (see [here](https://github.com/Eloffredo/3DLatticeDimers)).
The notebook contains details on 
  - BM training,
  - artificial generation of new sequences after inference via Gibbs Sampling,
  - feature extraction for contact predictions.

The contact map of $10000$ distinct 3D Lattice structure is used to validate contact predictions and contains structures in the form ``[Stucture ID, x_coord, y_coord]``.
