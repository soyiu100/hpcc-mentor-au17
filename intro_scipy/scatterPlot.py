# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:24:58 2017

@author: Isaac Pang
"""

import pandas as pan
import numpy as n
import matplotlib.pyplot as pl

time = 6;
dates = pan.date_range(start='20170101', freq = '365D', periods=time)
data_set = n.zeros((time,3))
data_set[:,0] = n.random.rand(time)*500 + 500
data_set[:,1] = n.random.rand(time)*50 + 50
data_set[:,2] = n.random.rand(time)*1999 + 18

#yearly_dat_comb = {}
#for_index = 0
#while not for_index >= time:
#    yearly_dat_comb[dates[for_index]] = data_set[for_index, :]
#    for_index += for_index + 1
#
#df = pan.DataFrame(yearly_dat_comb)
#yrs_grp = list(yearly_dat_comb.keys())
#num_grp1 = list(yearly_dat_comb.values())[0]

# df.plot.scatter(yrs_grp, num_grp1, index=dates, columns=['Rats', 'Snakes', 'Peanut Butter'])

df = pan.DataFrame(data_set, index=[1,2,3,4,5,6], columns=['a', 'b', 'c'])
df['dates'] = pan.Series([0,1,2,3,4,5])
ax = pl.gca()
df.plot.scatter(x='dates',y='a',ax=ax)
df.plot.scatter(x='dates',y='b',ax=ax)
df.plot.scatter(x='dates',y='c',ax=ax)

