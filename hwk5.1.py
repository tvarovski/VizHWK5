import numpy as np
import pandas as pd
data = pd.read_csv('C:\\Users\\Yihe\\Desktop\\Output.csv', index_col = 0)

"""
First, let's have a glance of which are the states containing the missing values. There are 8 states in total, namely, 
'California', 'Colorado', 'Georgia', 'Hawaii', 'Indiana', 'Louisiana','Minnesota', and 'Oklahoma'. Then we fill the missing 
values in different states with distinctive strategies.
"""
data[data.isnull().any(axis=1)].axes[0].unique().tolist()

['California',
 'Colorado',
 'Georgia',
 'Hawaii',
 'Indiana',
 'Louisiana',
 'Minnesota',
 'Oklahoma']


"""
Since we only have the divorce rate of 1990 in California, "backfill" method is the optimal one because the last-known 
data of divorce rate is available at the following time point. Similarly, the "backfill" strategy can also be use in 
refilling the missing data of Georgia, Hawaii, and Minnesota since they all share the same missing pattern. 
"""
data.loc[["California", "Georgia", "Hawaii", "Minnesota" ]].head(20)

data.loc[["California", "Georgia", "Hawaii", "Minnesota" ]] = data.loc[["California", "Georgia", "Hawaii", "Minnesota" ]].fillna(method = "backfill")

"""
For Colorado, since only one value was missing, it is sufficient to refill the value by the mean of the preceding 
and following years. There is no single value for the Divorce rate of Indiana, so the best way to refill the blank is to use the overall mean.
"""

data.loc[["Colorado"]]
data.loc[["Colorado"]] = data.loc[["Colorado"]].fillna(data.loc[["Colorado"]].iloc[12:16].mean())
data.loc[["Indiana"]] = data.loc[["Indiana"]].fillna(data.mean())


"""
It is sufficient to use the forward refilling strategy to refill the NA of Oklahoma. For Louisiana, both the 
forward and the backward refilling strategies are used. 
"""

data.loc[["Oklahoma"]] = data.loc[["Oklahoma"]].fillna(method = "pad")
data.loc[["Louisiana"]] = data.loc[["Louisiana"]].fillna(method = "pad").fillna(method = "bfill")

#Finally, no missing value in the dataframe is found.
data.isnull().values.any()

data.to_csv('C:\\Users\\Yihe\\Desktop\\refilled_data.csv', sep=',',encoding='utf-8')
