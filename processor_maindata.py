import pandas as pd
import numpy as np
import re
from sklearn import ensemble, preprocessing
from collections import Counter
from datetime import datetime
import catrates

# identifier columns
# Date                       object

numerical_features = ['Latitude', 'Longitude', 'Block', 'Species','Trap','week']
categorial_features = []#['Species_rr','Trap_rr'] 

# not used
# Address                    object
# Street                     object
# AddressNumberAndStreet     object
# AddressAccuracy             int64
# NumMosquitos                int64

features = numerical_features + categorial_features

missing_value_default = -999


def encode_categorical_features(train, test, cat_feature):
    '''
    encoded categorical features
    '''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(train[cat_feature].tolist() + test[cat_feature].tolist())
    train[cat_feature] = lbl.transform(train[cat_feature].values)
    test[cat_feature] = lbl.transform(test[cat_feature].values)
    return train, test

def encode_trap(train, test):
    '''
    encoded Trap as just the numerical value of the trap number
    '''
    train['Trap'] = train['Trap'].apply(lambda x: int(re.sub(r'[A-Z]','',x)))
    test['Trap'] = test['Trap'].apply(lambda x: int(re.sub(r'[A-Z]','',x)))
 
    return train, test


def transform_data(df,target=None):
    
    
    # change from object to datetime, numeric
    df['Date'] = pd.to_datetime(df['Date'])

    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # combine records that split due to over 50
    if target:
        dfg = df.groupby(['Date','Trap','Species'], as_index = False)\
                        .agg({'WnvPresent':{'WnvPresent':max},
                              'NumMosquitos':{'NumMosquitos':sum},
                              'Latitude':{'Latitude':max},
                              'Longitude':{'Longitude':max},
                              'Block':{'Block':max}
                            })
        dfg.columns = dfg.columns.get_level_values(0)

    else:
        dfg = df[['Date','Trap','Species', 'Latitude','Longitude','Block']].copy()

        
    # derived features
    dfg['week'] = dfg['Date'].apply(lambda x: x.week)
    dfg['month'] = dfg['Date'].apply(lambda x: x.month)
    dfg['year'] = dfg['Date'].apply(lambda x: x.year)


    return dfg

#    if target:
#        return dfg[['Date','Trap',target] + features]
#    else:
#        return dfg[['Date','Trap'] + features]



def merge_weather(df, weather):
    w = weather[weather['Station']==1] 
    #return pd.merge(df,w, on='Date')
    return pd.merge(df,w, on='week')


def merge_spray(df, spray):
    df_out = pd.merge(df,spray, on='week', how='left')
    df_out.fillna(missing_value_default, inplace=True)
    return df_out


def compute_categorical_rates(train, target):
    
    overall_mean = train[target].mean()
    rrs_out = {}

    for feature in categorial_features:
        merge_col = feature.replace('_rr','')
        rrs_out.update({merge_col : 
                        catrates.compute_categorical_rates(train, merge_col, target)})

    return overall_mean, rrs_out


def merge_categorical_rates(df, overall_mean, rrs):

    # main data
    for feature in categorial_features:
        merge_col = feature.replace('_rr','')
        df = pd.merge(df,rrs[merge_col], on=merge_col, how='left')
        df[feature] = df[feature].fillna(overall_mean)
    # weather
    return df

