#!/usr/bin/python

import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import operator
import time
import numpy as np
import pandas as pd

sys.path.append("../tools/")

# define the features
money_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                  'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
mail_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
feature_use = ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses',
               'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person',
               'from_this_person_to_poi', 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# analyze the data
print "sum of data =", len(data_dict)
is_poi = []
not_poi = []
feature_nan_dic = dict()
name_nan_dic = dict()
for k, v in data_dict.iteritems():
    if v['poi'] is True:
        is_poi.append(1)
    elif v['poi'] is False:
        not_poi.append(0)

    if len(feature_nan_dic) == 0:
        for feature in v:
            feature_nan_dic[feature] = 0

    name_nan_dic[k] = 0
    for feature in v:
        if v[feature] == 'NaN':
            feature_nan_dic[feature] += 1
            name_nan_dic[k] += 1

print feature_nan_dic
print name_nan_dic
print "sum of poi =", len(is_poi)
print "sum of not poi =", len(not_poi)

# clean the outlier
data_dict.pop("TOTAL", 0)
number_of_feature = len(feature_use)

pop_name = []
for k, v in data_dict.iteritems():
    count = 0
    for item in feature_use:
        if v[item] == 'NaN':
            v[item] == 0.0
            count += 1
    loss = float(count) / number_of_feature
    if loss >= 0.90:
        pop_name.append(k)

pop_name.append('THE TRAVEL AGENCY IN THE PARK')
print pop_name
print len(pop_name), "people have been avoided"
for name in pop_name:
    data_dict.pop(name, 0)

print "Lens of data_dict =", len(data_dict)

# generate new feature data
new_feature_list = []
for k,v in data_dict.iteritems():
    if v['from_this_person_to_poi'] == 'NaN' or v['total_stock_value'] == 'NaN':
        new_feature_list.append(0)
        continue
    elif v['from_this_person_to_poi'] == 0:
        new_feature_list.append(0)
        continue
    new_feature_list.append(1.0 * v['total_stock_value']/v['from_this_person_to_poi'])

data_newfeature = np.zeros((142, 1))
for i in range(142):
    data_newfeature[i,0] = new_feature_list[i]

# choose feature
feature_list_raw = np.array(["poi"] + feature_use)
data = featureFormat(data_dict, feature_list_raw)

data_test = np.zeros((142, 15))
data_test[:,:-1] = data
for i in range(142):
    data_test[i,14] = new_feature_list[i]

poi, features = targetFeatureSplit(data_test)
features_train, features_test, poi_train, poi_test = train_test_split(features, poi, test_size=0.2, random_state=42)

# Test new feature
newfeatures_train, newfeatures_test, newpoi_train, newpoi_test = train_test_split(data_newfeature, poi, test_size=0.2, random_state=42)
clf_nb_for_new_feature = GaussianNB()
t0 = time.time()
clf_nb_for_new_feature.fit(newfeatures_train, poi_train)
print "training time =", round(time.time() - t0, 3), "s"
score_NB_for_new_feature = clf_nb_for_new_feature.score(newfeatures_test, poi_test)
print "score =", score_NB_for_new_feature, "using new feature"


# k is how many feature I used expected poi
feature_number = 3
selector = SelectKBest(f_classif, k=feature_number)
selector.fit(features_train, poi_train)

features_train_selected = selector.transform(features_train)
features_test_selected = selector.transform(features_test)

scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train_selected)
rescaled_features_test = scaler.fit_transform(features_test_selected)
print selector.scores_

# get the my_features_list for dumping at the bottom of the code
score_index = 0
choose_feature = dict()
# choose_feature_test = dict(selector.scores_)
for score in selector.scores_:
    choose_feature[score_index] = score
    score_index += 1
choose_feature_itemgetter = sorted(choose_feature.items(), key=operator.itemgetter(1), reverse=True)

feature_count = 0
my_features_list = ['poi']
for k, _ in choose_feature_itemgetter:
    # TOP 3 used
    if feature_count < features_train_selected.shape[1]:
        my_features_list.append(feature_use[k])
        feature_count += 1

print my_features_list, "is using"

# classifier 1 - GaussianNB
clf_nb = GaussianNB()
t0 = time.time()
clf_nb.fit(features_train_selected, poi_train)
print "training time =", round(time.time() - t0, 3), "s"
score_NB = clf_nb.score(features_test_selected, poi_test)
print "score =", score_NB, "using GaussianNB"

preds = clf_nb.predict_proba(features_test_selected)[:, 1]
fpr, tpr, _ = roc_curve(poi_test, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
auc = auc(fpr,tpr)

# classifier 2 - DecisionTree
parameters_dt = {'max_depth': range(12, 20)}
clf_decisiontree = GridSearchCV(DecisionTreeClassifier(), parameters_dt, cv=4, scoring='recall')
clf_decisiontree.fit(features_train_selected, poi_train)
score_decisiontree = clf_decisiontree.score(features_test_selected, poi_test)
print "score =", score_decisiontree, "using decision tree"
print "parameters =", clf_decisiontree.best_params_

# classifier 3 - RandomTree
parameters_rf = {'max_depth': range(1, 5)}
clf_randomtree = GridSearchCV(RandomForestClassifier(), parameters_rf)
clf_randomtree.fit(features_train_selected, poi_train)
score_randomtree = clf_randomtree.score(features_test_selected, poi_test)
print "score =", score_randomtree, "using random tree"
print "parameters =", clf_randomtree.best_params_

# classifier 4 - PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

estimators = [('pca', PCA()), ('svm', svm.SVC())]
parameter_pca = {'pca__n_components': range(2, 3),
                 'svm__kernel': ['rbf', 'poly', 'linear'],
                 'svm__C': [1, 10, 100],
                 'svm__gamma': [0.001, 0.005, 0.01]}

pl = Pipeline(estimators)
clf_pca = GridSearchCV(pl, parameter_pca, cv=5)
clf_pca.fit(rescaled_features_train, poi_train)
print 'Best_estimator = {0}'.format(clf_pca.best_estimator_.get_params())
score_pca = clf_pca.score(rescaled_features_test, poi_test)
print "score =", score_pca, "using PCA"

# classifier 5 - SVM
parameters_svm = {'kernel': ['rbf', 'linear'], 'C': [1, 10, 100], 'gamma': [0.0005, 0.001, 0.005]}
svc = svm.SVC()
clf_svm = GridSearchCV(svc, parameters_svm, cv=5, scoring='recall')

clf_svm.fit(rescaled_features_train, poi_train)
print "training time =", round(time.time() - t0, 3), "s"
score_svm = clf_svm.score(rescaled_features_test, poi_test)
print "score =", score_svm, "using svm"
print "parameters =", clf_svm.best_params_

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys=True)

# Use KFold method to cross check the classifier
t0 = time.time()
from sklearn.model_selection import KFold
kf = KFold(4, shuffle=True)
score_kf = []
accuracy_kf = []

for train_indices, test_indices in kf.split(features):
    # make training and testing dataset
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    poi_train = [poi[ii] for ii in train_indices]
    poi_test = [poi[ii] for ii in test_indices]

    selector = SelectKBest(f_classif, k=3)
    selector.fit(features_train, poi_train)

    features_train_selected = selector.transform(features_train)
    features_test_selected = selector.transform(features_test)

    t0 = time.time()
    clf_nb.fit(features_train_selected, poi_train)
    print "training time =", round(time.time() - t0, 3), "s"
    score_NB = clf_nb.score(features_test_selected, poi_test)
    score_kf.append(score_NB)

    from sklearn.metrics import accuracy_score
    pred_nb = clf_nb.predict(features_test_selected)
    accu = accuracy_score(poi_test, pred_nb)
    accuracy_kf.append(accu)

print "score =", 1.0 * sum(score_kf)/len(score_kf)
print "accuracy =", 1.0 * sum(accuracy_kf)/len(accuracy_kf)

# Dump my classifier
dump_classifier_and_data(clf_nb, my_dataset, my_features_list)
