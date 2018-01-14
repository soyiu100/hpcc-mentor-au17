# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 07:29:37 2017

@author: Isaac Pang
"""

import pandas as pan
import numpy as n

time = 30;
dates = pan.date_range(start='20170101', freq = '365D', periods=time)
data_set = n.zeros((time,3))
data_set[:,0] = n.random.rand(time)*500 + 500
data_set[:,1] = n.random.rand(time)*50 + 50
data_set[:,2] = n.random.rand(time)*1999 + 18

df = pan.DataFrame(data_set, index=dates, columns=['Rats', 'Snakes', 'Peanut Butter'])
ax = df.plot.bar()
# alternatively, df.plot(kind = "bar") works