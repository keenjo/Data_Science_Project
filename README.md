# Data_Science_Project
Data Science Project for M1 TAL at Université de Lorraine
## Structure of the Repo:
- `requirements.txt` contains a list of all of the libraries required to run all of the scripts in this repository
  - these required libraries can be installed by running `pip install -r requirements.txt`
- `scraping_functions.py` contains the functions used to collect all of the data
- `preprocessing.py` contains the functions used to preprocess the scraped data
- The `data` folder contains all of the data
  - `scraped_data.json` is a json structured file containing the scraped data before preprocessing
  - `preprocessed_data.json` is a json structured file containing the scraped data after preprocessing
- `clustering.py` contains functions for clustering
- `cluster_results` contains results from a few different tests on our clustering script
  - The last part of the name of each subfolder in this folder indicates the feature/features used for clustering
- `classification.py` contains functions for classification

## Information on use of the repository:

### Running the code
- Before running any of the scripts in this project we recommend that you open the scripts in order to adjust any of the parameters if you would like to change them from their default values. This should be fairly straightforward as we have placed any adjustable parameters at the top of every script with an explanation for each parameter. Once you have adjusted the parameters to your liking, you can easily run any of the scripts through the command line.

### Finding the outputs
- Outputs from the `scraping_functions.py` and `preprocessing.py` scripts can be found in the `data` folder that will be created when you run those scripts
- Each output from the `clustering.py` script can be found in its own folder named via the following format `clustering_results_[feature used for clustering]`
