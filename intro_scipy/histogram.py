# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:08:52 2017

@author: Isaac Pang
"""
import numpy as n
import matplotlib.pylab as pl

data = n.random.rand(100)*100
xaxis= n.arange(0, 110, 5)
pl.hist(data, rwidth=9/10, bins=xaxis, align='mid', color='green')
pl.xticks(xaxis)
pl.xscale('linear', basex=5)
pl.xlabel('#s Produced')
pl.ylabel('Frequency')
pl.title('Histogram of Random Numbers')
pl.show()
