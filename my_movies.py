# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:54:36 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("my_movies.csv")

df.shape
df.dtypes
df.info()
df.head()
df = df.drop(df.columns[:5],axis=1)
df

df.values[0]
df.values[1]

# Apply apriori algorithm
from mlxtend.frequent_patterns import apriori
frequent_moviesets = apriori(df,min_support=0.01,use_colnames=(True))
frequent_moviesets 

# i have applyed different min_support values then i have taken 0.01 value

from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_moviesets,metric="confidence",min_threshold=0.7)
pd.set_option("display.max_columns",9)
res

# i have applyed different min_threshold values then i have taken 0.7 value

res1 = res[["antecedents","consequents","support","confidence","lift"]]
res1
# showing the results of highest lifting values
res1.nlargest(10, columns="lift")

# therefor the person who is seen (Green MIle and Gladiators) Movies he has 10 times more chance to see (LOTR ) Movie also, like that the result is showing top 10 probability. 












