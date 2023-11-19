import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime,time

from bokeh.io import output_notebook, show
from bokeh.plotting import figure

#스캐터
p = figure(plot_width=600,plot_height=400)
x=list(range(0,7))
y=list(range(0,7))
p.circle(x,y,size=10,line_color="orange",fill_color="blue")
show(p)

