# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:48:56 2017

@author: Isaac Pang
"""

######
"""
To enable interactive plots, in the ipython console, type
%matplotlib
"""

import numpy as n
import matplotlib.pyplot as pl 

x = n.linspace(-2*n.pi, 2*n.pi, 100)
y = n.sin(1/x)

axis = pl.gca()
axis.spines['bottom'].set_position(('data',0))
axis.spines['top'].set_position(('data',0))
axis.spines['right'].set_position(('data',0))
axis.spines['left'].set_position(('data',0))

pl.plot(x,y)
pl.show()