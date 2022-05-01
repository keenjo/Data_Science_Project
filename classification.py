import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.model_selection
import sklearn.naive_bayes

# Parameters for reproducibility of the results
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.25
MAX_FEATURES = 400
ALPHA = 0.015 # Computed using GridSearchCV (see end of the file)


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
df = pd.read_json('preprocessed_data.json')
df = sklearn.utils.shuffle(df, random_state=RANDOM_STATE)

# Stores category names for each category index
idx_to_category = {row['category number']: row['category'] for index, row in df.iterrows()}

vectorizers = []
X = []


# Main training loop. We train a different vectorizer (with different parameters) for each meta-feature.
# The input X of the perceptron is created by stacking the individual inputs X for each meta-features.
for column_name in selected_features:
    (use_idf, max_features) = feature_options[column_name]
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_features=max_features, use_idf=use_idf, stop_words='english')
    vectorizers.append(vectorizer)
    
    # TfidfVectorizer only accepts lists of strings as input (and not lists of lists of words)
    texts = [' '.join(row) for row in df[column_name].to_list()]
    
    # Create the partial input for the current meta-feature
    X_partial = vectorizer.fit_transform(texts)
    X.append(X_partial)
    

# Create the final input
X = scipy.sparse.hstack(X)
Y = df['category number']

perceptron = sklearn.linear_model.Perceptron(random_state=RANDOM_STATE, alpha=ALPHA)


# Split 'n fit
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)
perceptron.fit(X_train, Y_train)


# Predict and display performance metrics
Y_pred = perceptron.predict(X_test)

print('Predictions:')
for pred in Y_pred:
    print(str(pred).ljust(3), end='')
print()

print('Expected:')
for pred in Y_test:
    print(str(pred).ljust(3), end='')
print()

# Accuracy
print('Accuracy: ', sklearn.metrics.accuracy_score(Y_test, Y_pred))

# Confusion matrix (opens in a graphical window)
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)

# Metrics for each class
print(sklearn.metrics.classification_report(Y_test, Y_pred))

plt.show()

# To display the top features for each class, we first need to map each feature index back to its corresponding feature name
idx_to_tag = dict()

for i, vectorizer in enumerate(vectorizers):
    # For each meta-feature, add the corresponding offset to the index
    for k, v in vectorizer.vocabulary_.items():
        idx_to_tag[feature_idx_offset[selected_features[i]] + v] = k

# Display top features
print('Category'.ljust(30), 'Meta-feature'.ljust(30), 'Feature'.ljust(20), 'Weight'.ljust(20))
print('-'*100)
for class_num, class_weights in enumerate(perceptron.coef_):
    # Compute and display the top features and weights for the class
    top_vals = sorted(enumerate(class_weights), reverse=True, key=lambda x: x[1])
    for id, val in top_vals[:10]:
        print(idx_to_category[class_num].ljust(30), idx_to_feature(id).ljust(30), idx_to_tag[id].ljust(20), val.round(3))


# GridSearchCV, used for finding the best `alpha` parameter for the perceptron.
parameters = [a for a in np.logspace(-5, 1, 50)]
clf = sklearn.model_selection.GridSearchCV(estimator=sklearn.naive_bayes.MultinomialNB(), 
                   param_grid={'alpha': parameters},
                   scoring='accuracy',
                   return_train_score=True,
                   cv=5
                  )

clf.fit(X_train, Y_train)
best_parameters = clf.best_estimator_.get_params()

cv_res = pd.DataFrame(clf.cv_results_)
#print(cv_res)

print("Best GridSearchCV score: %0.3f" % clf.best_score_)
print("Best GridSearchCV alpha", best_parameters['alpha'])
