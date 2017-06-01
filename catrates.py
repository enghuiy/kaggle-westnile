import pandas as pd
import numpy as np

#SE_RATIO = 0.20
PSEUDOCOUNT = 0.0001

def se(p,n):
    '''
    standard error of a benoulli trial with mean p, sampled n times
    '''
    if n == 0:
        return 0
    return np.sqrt(p*(1-p)/n)

def mean_pseudocount(ds):
    total=sum(ds)
    count=len(ds)
    return (total+PSEUDOCOUNT)/(count+PSEUDOCOUNT)


def compute_categorical_rates(df, col_name, target):
    '''
    check if sample size is sufficient for S.E. to be less than x%
    S.E. = sqrt(p*q/n)
    if not, use overall mean for that category
    '''
    overall_mean = df[target].sum()/len(df)
    rr_col_name = col_name+'_rr'

    x = df.groupby(col_name,as_index=False)[[target]].\
    agg({target:{'mean':mean_pseudocount,'hit':sum,'count':len}}).sort_values('count',ascending=False)
    x['SE']=x.apply(lambda x: se(overall_mean,x['count']), axis=1)
    x['ratio']=x['SE']/x['mean']
    #x[rr_col_name] = x.apply(lambda x: x['mean'] if x['ratio']<=SE_RATIO else overall_mean, axis=1)
    x[rr_col_name] = x.apply(lambda x: x['mean'] if x['hit']>=10 else overall_mean, axis=1)
    return x #[[col_name, rr_col_name]]

    
