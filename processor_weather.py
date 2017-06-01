import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from collections import Counter
from datetime import datetime

numerical_features = ['Tmax','Depart','DewPoint', 'WetBulb', 'Heat','Cool',
                      'Sunrise','Sunset',  
                      'StnPressure', 'AvgSpeed', 
                      'BR','RA','TS','DZ','FG',
                      'PrecipTotal', 'ResultSpeed','Tavg',
                      'Tmin','SeaLevel'
                      ]

categorial_features = [] #['CodeSum_rr']

features = numerical_features + categorial_features

missing_value_default = -999


def transform_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    
    # missing values
    missing_patterns = ['M', '-','T', ' T', '  T']
    for pattern in missing_patterns:
        df.replace(pattern, missing_value_default, inplace=True)
    
    # change from object to numeric
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    
    # encode the most common codesums
    df['RA'] = df['CodeSum'].str.contains('RA').astype(int)
    df['TS'] = df['CodeSum'].str.contains('TS').astype(int)
    df['DZ'] = df['CodeSum'].str.contains('DZ').astype(int)
    df['BR'] = df['CodeSum'].str.contains('BR').astype(int)
    df['FG'] = df['CodeSum'].str.contains('FG').astype(int)
 
    # derived features
    df['week'] = df['Date'].apply(lambda x: x.week)
    df['dayofweek'] = df['Date'].apply(lambda x: x.dayofweek)

    # group by week and only take mon,tues/wed readings
    def mean_omit_missing(ds):
        return np.mean([x for x in ds if x != -999])
    dfg = df[(df['dayofweek']<3)][['week','Station']+numerical_features]\
        .groupby(['week','Station'], as_index=False)\
        .agg(mean_omit_missing)
    dfg.columns = dfg.columns.get_level_values(0)

    return dfg
    #return df[['Date','Station']+features]
