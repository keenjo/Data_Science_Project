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
- `classif_results` contains results and scores obtained during classification

## Information on use of the repository:

### Running the code
- Before running any of the scripts in this project we recommend that you open the scripts in order to adjust any of the parameters if you would like to change them from their default values. This should be fairly straightforward as we have placed any adjustable parameters at the top of every script with an explanation for each parameter. Once you have adjusted the parameters to your liking, you can easily run any of the scripts through the command line.

### Finding the outputs
- Outputs from the `scraping_functions.py` and `preprocessing.py` scripts can be found in the `data` folder that will be created when you run those scripts.
- Each output from the `clustering.py` script can be found in its own folder named via the following format `cluster_results/clustering_results_[feature used for clustering]`.
- The output of the `classification.py` script can be found in its own folder named `classif_results`.

### Bonus Points
- We believe that we would be eligible to receive bonus points for a few different tests and features that we have provided which were not required:
  - For the `preprocessing` we have provided additional information via the extraction and processing of Named Entity Recognition (NER), Part-of-Speech (POS), and lemmas from the article text.
  - For the `clustering`, other than providing the required visualization of the intrinsic and extrinsic evaluation scores we have also provided visualization for the clustering scatterplot and inertia. On top of this, we also provided a visualization of how the evaluation scores change with respect to a change in the number of TFIDF features and we implemented a function to extract the top feature terms from each cluster. Lastly, we tested our clustering on several different features and enabled the ability to stack multiple different types of features into one TFIDF matrix, so the program is not limited to only using one feature for clustering. The results of these experiments can be found in `cluster_results/`. We found that the experiments using triples as well as a combination of triples and lemmas performed best of all the clustering experiments.
  - For the `classification` we have tested our script on several different sets of features as well as several different combinations of sets of features via a similar method used in our clustering script of stacking multiple types of features into one TFIDF matrix. Furthermore, we have implemented a script which is able to test nine different classification methods and all of which have been tested: Perceptron, K Neighbors Classifier, Support Vector Machine, Decision Tree Classifier, Random Forest Classifier, Multi-layer Perceptron Classifier, Ada Boost Classifier, Gaussian Naive Bayes, and Quadratic Discriminant Analysis. We have also conducted extensive hyperparameter tuning through GridSearchCV, obtaining a final test accuracy of 98.97%.
