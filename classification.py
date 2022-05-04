print('Importing libraries...')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import re
import warnings

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=UserWarning)

# Parameters for reproducibility of the results
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.25
MAX_FEATURES = 400

# Meta-features selected for training. Possible values are:
# 'title', 'text', 'processed text', 'description', 'processed description', 'POS',
# 'nouns', 'verbs', 'NER', 'NER tokens', 'lemmas', 'triples', 'processed triples'
selected_features = [
    'processed text',
    'processed description',
    'nouns',
    'verbs',
    'NER tokens',
    'processed triples'
]

# Parameters for each of the meta-features.
# We ended up using `use_idf=False` everywhere since it surprisingly gave the best results.
feature_options = {
    # Column name            use_idf  max_features
    'processed text':        (False,   500),
    'processed description': (False,   400),
    'nouns':                 (False,   100),
    'verbs':                 (False,   100),
    'NER tokens':            (False,   50),
    'processed triples':     (False,   400),
}

# List of classifiers and parameter spaces to try.
# Please note that the initial search space was larger, but that we have reduced it
# so that the training wouldn't last too long.
print('Loading parameter grid...')
classifiers = [
    (Perceptron(random_state=RANDOM_STATE), {
        'alpha': np.logspace(-5, 1, 20),
        'max_iter': [5, 10, 20, 50, 100]
    }),
    (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 8, 12, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }),
    (SVC(random_state=RANDOM_STATE), {
        'C': np.logspace(-2, 2, 10),
        'gamma': np.logspace(-3, 3, 10),
        'kernel': ['rbf', 'poly', 'sigmoid']
    }),
    (DecisionTreeClassifier(random_state=RANDOM_STATE), {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 5)
    }),
    (RandomForestClassifier(random_state=RANDOM_STATE), {
        'criterion' :['gini', 'entropy'],
        'n_estimators': list(map(int, np.logspace(1, 3, 7))),
        'max_features': ['sqrt', 'log2'],
        'max_depth': range(2, 10)
    }),
    (MLPClassifier(random_state=RANDOM_STATE), {
        'solver': ['lbfgs'],
        'max_iter': list(map(int, np.logspace(2, 3.5, 4))),
        'alpha': np.logspace(-10, -1, 30),
        'hidden_layer_sizes': list(map(int, 2**np.linspace(1, 8, 8)))
    }),
    (AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, max_features="auto", class_weight="balanced")), {
        'base_estimator__max_depth': range(2, 10),
        'base_estimator__min_samples_leaf': [5, 10],
        'n_estimators': list(map(int, np.logspace(1, 3, 7))),
        'learning_rate': np.logspace(-5, -1, 20)
    }),
    (GaussianNB(), {
        'var_smoothing': np.logspace(-10, -1, 30)
    }),
    (QuadraticDiscriminantAnalysis(), {
        'reg_param': np.linspace(0.1, 0.5, 5)
    }),
]


def idx_to_feature(idx):
    '''
        Maps a feature index to the name of the corresponding meta-feature.
        Arguments:
            `idx` the index of the feature
        Returns:
            The name of the corresponding meta-feature.
    '''
    feature_counter = 0
    for feature in selected_features:
        feature_counter += feature_options[feature][1]
        if feature_counter > idx:
            return feature


# Stores the index of the first feature for each meta-feature
feature_idx_offset = dict()
feature_counter = 0
for feature in selected_features:
    feature_idx_offset[feature] = feature_counter
    feature_counter += feature_options[feature][1]


# Read and shuffle input file
print('Reading input file...')
df = pd.read_json('data/preprocessed_data.json')
df = shuffle(df, random_state=RANDOM_STATE)

# Stores category names for each category index
idx_to_category = {row['category number']: row['category'] for index, row in df.iterrows()}

vectorizers = []
X = []

print('Running input through TfidfVectorizer...')

# Vectorizer training loop. We train a different vectorizer (with different parameters) for each meta-feature.
# The input X of the perceptron is created by stacking the individual inputs X for each meta-features.
for column_name in selected_features:
    (use_idf, max_features) = feature_options[column_name]
    vectorizer = TfidfVectorizer(max_features=max_features, use_idf=use_idf, stop_words='english')
    vectorizers.append(vectorizer)
    
    # TfidfVectorizer only accepts lists of strings as input (and not lists of lists of words)
    texts = [' '.join(row) for row in df[column_name].to_list()]
    
    # Create the partial input for the current meta-feature
    X_partial = vectorizer.fit_transform(texts)
    X.append(X_partial)

# Create the final input
X = scipy.sparse.hstack(X)
Y = df['category number']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)

best_classifiers = []
display_strings = []

print('Optimizing hyperparameters for all models...')
print('Warning: This operation can take a long time (~1 hour). Results will be displayed at the end.')

# Classifier optimization loop
for i, (classifier, parameters) in enumerate(classifiers):

    # For each classifier, we first create a GridSearchCV object for the desired parameters
    clf = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', return_train_score=True, cv=5, verbose=2)

    # Some classifiers don't accept sparse matrices as input
    X_model_train = X_train
    if isinstance(classifier, (GaussianProcessClassifier, GaussianNB, QuadraticDiscriminantAnalysis)):
        X_model_train = X_model_train.toarray()
    
    # Compute the optimal hyperparameters
    clf.fit(X_model_train, Y_train)
    best_parameters = clf.best_estimator_.get_params()
    
    # Store best classifier and accuracy
    best_classifiers.append((clf.best_score_, clf.best_estimator_))
    
    display_strings.append([
        str(classifier.__class__).ljust(35),                          # Classifier name
        str({k: best_parameters[k] for k in parameters}).ljust(100), # All optimized parameters
        clf.best_score_                                              # Accuracy
    ])
    print(f'Optimizing hyperparameters for all models ({i+1}/{len(classifiers)})...')

print()
print('Classifier'.ljust(35), 'Best parameters'.ljust(100), 'Best accuracy')
print('-'*145)

for s1, s2, s3 in display_strings:
    print(s1, s2, s3)

# Sort classifiers by accuracy and pick the best one
best_accuracy, best_classifier = sorted(best_classifiers, reverse=True)[0]

Y_pred = best_classifier.predict(X_test.toarray())

print('Best classifier', best_classifier)
print('Training accuracy', best_accuracy)
print('Test accuracy', accuracy_score(Y_test, Y_pred))
print()

# Show predicted vs expected classes
print('Predictions:')
for pred in Y_pred:
    print(str(pred).ljust(3), end='')
print()

print('Expected:')
for pred in Y_test:
    print(str(pred).ljust(3), end='')
print()

# Confusion matrix (opens in a graphical window)
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)

print(classification_report(Y_test, Y_pred))

plt.show()

# To display the top features for each class, we first need to map each feature index back to its corresponding feature name
idx_to_tag = dict()

for i, vectorizer in enumerate(vectorizers):
    # For each meta-feature, add the corresponding offset to the index
    for k, v in vectorizer.vocabulary_.items():
        idx_to_tag[feature_idx_offset[selected_features[i]] + v] = k

# # Display top features
# print('Category'.ljust(30), 'Meta-feature'.ljust(30), 'Feature'.ljust(20), 'Weight'.ljust(20))
# print('-'*100)
# for class_num, class_weights in enumerate(perceptron.coef_):
    # # Compute and display the top features and weights for the class
    # top_vals = sorted(enumerate(class_weights), reverse=True, key=lambda x: x[1])
    # for id, val in top_vals[:10]:
        # print(idx_to_category[class_num].ljust(30), idx_to_feature(id).ljust(30), idx_to_tag[id].ljust(20), val.round(3))

