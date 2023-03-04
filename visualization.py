import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

def line_points(x, slope, intercept):
    return x, [slope*i + intercept for i in x]

def lreg_plot(reactions, slope, intercept):
    fig, ax = plt.subplots()
    ## scatter plot 
    markers = ["o","v","^","<",">","1","2","3","4","s","p","P","*","+","x","X","D","_"]
    subjects = reactions.Subject.unique()
    groups = reactions.groupby('Subject')
    for name,group in groups:
        idx = list(subjects).index(name)
        ax.plot(group.Days, group.Reaction, c = 'navy', marker = "o", linestyle = '')
        ax.plot(group.Days, group.Reaction, c = 'lightblue')
    ## reg line
    x,y = line_points(reactions.Days, slope, intercept)
    ax.plot(x , y, c = "blue", linestyle = '-', linewidth = 3)  
    ##
    ax.set_xlabel("Days of Driving")
    ax.set_ylabel("Reaction Time (ms)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def kruschke_style_posteriors(trace):
    ax = az.plot_posterior(trace, hdi_prob=0.97, textsize=20)
    fig = ax.ravel()[0].figure
    fig.savefig('reaction_posterior_arviz.pdf', dpi=300, bbox_inches="tight", transparent=True)