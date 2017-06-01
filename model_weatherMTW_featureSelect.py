
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from datetime import datetime
import processor_weather
import processor_maindata

IMPORTANCE_THRESHOLD = 0.01

def build_model(train,test,features,target,classifier_to_use,param_distributions,cv=None):
    
    '''
    build a model using RFC (default) or GBC
    - automated cross-validation search of hyperparameters
    - automated feature selection
    '''
    if cv == None:
        cv = 3

    if classifier_to_use=='GBC':
        classifier = ensemble.GradientBoostingClassifier()
    else:
        classifier = ensemble.RandomForestClassifier()
    
    # optimal hyperparameters (using GridSearchCV)
    myCV = GridSearchCV(classifier, param_distributions,
                        scoring='roc_auc', n_jobs=-1, cv=cv, verbose=0)

    myCV.fit(train[features], y=train[target])

    clf = myCV.best_estimator_
    print ("best params: ",myCV.best_params_)

    print ("Feature Importance (all features): ")
    for feature, importance in sorted(zip(features,clf.feature_importances_), key=lambda x:x[1],reverse=True):
        print (feature, importance)
    
    # feature selection
    sfm = SelectFromModel(clf,threshold=IMPORTANCE_THRESHOLD)
    X_train_transform = sfm.fit_transform(train[features],train[target])
    X_test_transform = sfm.transform(test[features])

    clf.fit(X_train_transform,train[target])
 
    model_statistics(clf, X_train_transform, train[target], X_test_transform,test[target])

    
    return myCV.best_params_, clf, sfm



def model_statistics(model,X_train,y_train, X_val, y_val):
    '''
    report model performance
    '''
    predict_scores = model.predict_proba(X_train)
    print ("train AUC: ", roc_auc_score(y_train, predict_scores[:,1]))
    
    predict_scores = model.predict_proba(X_val)
    print ("test AUC: ", roc_auc_score(y_val, predict_scores[:,1]))
    return


# Load dataset 
train_in = pd.read_csv('../input/train.csv')
test_in = pd.read_csv('../input/test.csv')
weather_in = pd.read_csv('../input/weather.csv')
#spray_in = pd.read_csv('../input/spray.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')

# parameters
target = 'WnvPresent'
MIN_HIT = 30 # for random forest min_leaf_samples


# encode categorical features
train_cat, test_cat = processor_maindata.encode_categorical_features(train_in, test_in, 'Species')
train_cat, test_cat = processor_maindata.encode_trap(train_cat, test_cat)

# process weather data
weather = processor_weather.transform_data(weather_in)


train_in_transformed = processor_maindata.transform_data(train_cat, target=target)
train_in_transformed = processor_maindata.merge_weather(train_in_transformed, weather)

test_transformed = processor_maindata.transform_data(test_cat)
test_transformed = processor_maindata.merge_weather(test_transformed, weather)

# split provided train into training and out_of_time test sets
test_year = 2013
mytrain =  train_in_transformed[train_in_transformed['year']!=test_year]
mytest =  train_in_transformed[train_in_transformed['year']==test_year]
print ('my training, test sets:', len(mytrain), len(mytest))

# prepare for training
features = processor_maindata.features + processor_weather.features 
print('features',features)
min_samples_leaf = int(MIN_HIT / mytrain[target].mean())
print ('min_samples_leaf',min_samples_leaf)

# random forest
#param_distributions={'n_estimators':[500],
#        'min_samples_leaf':[min_samples_leaf,int(1.5*min_samples_leaf)],
#        'max_features':['sqrt','log2',None]
#        }
#best_params, clf = build_model(mytrain,mytest,features,target,'RFC',param_distributions,5)

# gradient boosting classifier
param_distributions={'n_estimators':[15,25,50],
        'learning_rate':[0.1,0.125],
        'min_samples_leaf':[int(0.5*min_samples_leaf), min_samples_leaf],
        'subsample':[.4,.5,.6]
        }
best_params, clf, sfm = build_model(mytrain,mytest,features,target,'GBC',param_distributions,5)


# retrain model with all training data
train = train_in_transformed
min_samples_leaf = int(MIN_HIT / train_in_transformed[target].mean())
print ('min_samples_leaf',min_samples_leaf)

#clf = ensemble.RandomForestClassifier(max_features=best_params['max_features'], 
#                                        n_estimators=best_params['n_estimators'],
#                                          min_samples_leaf=best_params['min_samples_leaf'])                                    )
clf = ensemble.GradientBoostingClassifier(subsample=best_params['subsample'],
                                          n_estimators=best_params['n_estimators'],
                                          min_samples_leaf=min_samples_leaf,
                                          learning_rate=best_params['learning_rate'])

X_train_transformed = sfm.transform(train[features])
clf.fit(X_train_transformed,train[target])

predict_scores = clf.predict_proba(X_train_transformed)
print ("final train AUC: ", roc_auc_score(train[target], predict_scores[:,1]))

# create predictions and submission file
test_transformed = processor_maindata.transform_data(test_cat)
test_transformed = processor_maindata.merge_weather(test_transformed, weather)
test = test_transformed
X_test_transformed = sfm.transform(test[features])

predictions = clf.predict_proba(X_test_transformed)[:,1]
sample['WnvPresent'] = predictions
sample.to_csv('predicted_proba.csv', index=False)




