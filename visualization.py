import pandas as pd
import numpy as np
import arviz as az

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

c1 = "blue"
c2 = "orange"
c3 = "limegreen"
c4 = "mediumvioletred"

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

def homogenous_models_outputs(trace, prior_predictive, posterior_predictive, slope, intercept, rt):
    fig, ax = plt.subplots(1, 4, sharey = False,figsize=(21, 5))
    ##
    az.plot_dist(trace['a'], color=c1, ax=ax[0])
    az.plot_dist(prior_predictive['a'], color=c3, ax=ax[0])
    ax[0].vlines(x = intercept, ymin = 0, ymax = 0.06, colors = c2)

    ##
    az.plot_dist(trace['b'], color=c1, ax=ax[1], label="Probabilistic Model Posterior")
    az.plot_dist(prior_predictive['b'], color=c3, ax=ax[1], label="Probabilistic Model Prior")
    ax[1].vlines(x = slope, ymin = 0, ymax = 0.31, colors = c2, label = 'Non-Probabilistic Model')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.55, 1.31))

    ##
    az.plot_dist(trace['s'], color=c1, ax=ax[2])
    az.plot_dist(prior_predictive['s'], color=c3, ax=ax[2])

    ##
    az.plot_dist(posterior_predictive['rt'], color=c1, ax=ax[3])
    az.plot_dist(prior_predictive['rt'], color=c3, ax=ax[3])
    ax[3].vlines(x = rt, ymin = 0, ymax = 0.007, colors = c2)

    ##
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)

    ax[0].set_ylabel("Probability Density")
    ax[1].set_ylabel("Probability Density")
    ax[2].set_ylabel("Probability Density")
    ax[3].set_ylabel("Probability Density")

    ax[0].set_xlabel("a")
    ax[1].set_xlabel("b")
    ax[2].set_xlabel("s")
    ax[3].set_xlabel("rt")

    plt.savefig('ch1_Fig2.pdf', dpi=300, bbox_inches="tight", transparent=True)

def hom_het_hi_reg_line(reactions, trace, trace_het, trace_hi, posterior_predictive, posterior_predictive_het, posterior_predictive_hi, slope, intercept):
    fig, ax = plt.subplots(3, 4, sharey = False,figsize=(22, 15))

    ##
    subs = ['308','309','332','351']#337
    subs_idx = [0,1,5,11]#9
    ##
    rt_hom_prob_ = np.array([posterior_predictive['rt'][:,[s*len(reactions.Days.unique())+day for s in range(len(reactions.Subject.unique()))]].flatten() for day in reactions.Days.unique()])
    rt_hom_prob = np.array([[rt_hom_prob_[:,draw] for draw in range(rt_hom_prob_.shape[1])]])

    ## a
    ## homogenous non-probabilistic
    ax[0][0].vlines(x = intercept, ymin = 0, ymax = 0.06, colors = c2)
    ax[0][1].vlines(x = intercept, ymin = 0, ymax = 0.06, colors = c2)
    ax[0][2].vlines(x = intercept, ymin = 0, ymax = 0.06, colors = c2)
    ax[0][3].vlines(x = intercept, ymin = 0, ymax = 0.06, colors = c2, label = "hom non-prob")

    ## homogenous probabilistic
    az.plot_dist(trace['a'], color=c1, ax=ax[0][0])
    az.plot_dist(trace['a'], color=c1, ax=ax[0][1])
    az.plot_dist(trace['a'], color=c1, ax=ax[0][2])
    az.plot_dist(trace['a'], color=c1, ax=ax[0][3], label = "hom prob")

    ## heterogenous
    az.plot_dist(trace_het['a'][:,subs_idx[0]], color=c3, ax=ax[0][0])
    az.plot_dist(trace_het['a'][:,subs_idx[1]], color=c3, ax=ax[0][1])
    az.plot_dist(trace_het['a'][:,subs_idx[2]], color=c3, ax=ax[0][2])
    az.plot_dist(trace_het['a'][:,subs_idx[3]], color=c3, ax=ax[0][3], label = "het prob")

    ## hierarchical
    az.plot_dist(trace_hi['a'][:,subs_idx[0]], color=c4, ax=ax[0][0])
    az.plot_dist(trace_hi['a'][:,subs_idx[1]], color=c4, ax=ax[0][1])
    az.plot_dist(trace_hi['a'][:,subs_idx[2]], color=c4, ax=ax[0][2])
    az.plot_dist(trace_hi['a'][:,subs_idx[3]], color=c4, ax=ax[0][3], label = "hi prob")

    ## b
    ## homogenous non-probabilistic
    ax[1][0].vlines(x = slope, ymin = 0, ymax = 0.31, colors = c2)
    ax[1][1].vlines(x = slope, ymin = 0, ymax = 0.31, colors = c2)
    ax[1][2].vlines(x = slope, ymin = 0, ymax = 0.31, colors = c2)
    ax[1][3].vlines(x = slope, ymin = 0, ymax = 0.31, colors = c2, label = "hom non-prob")

    ## homogenous probabilistic
    az.plot_dist(trace['b'], color=c1, ax=ax[1][0])
    az.plot_dist(trace['b'], color=c1, ax=ax[1][1])
    az.plot_dist(trace['b'], color=c1, ax=ax[1][2])
    az.plot_dist(trace['b'], color=c1, ax=ax[1][3], label = "hom prob")

    ## heterogenous
    az.plot_dist(trace_het['b'][:,subs_idx[0]], color=c3, ax=ax[1][0])
    az.plot_dist(trace_het['b'][:,subs_idx[1]], color=c3, ax=ax[1][1])
    az.plot_dist(trace_het['b'][:,subs_idx[2]], color=c3, ax=ax[1][2])
    az.plot_dist(trace_het['b'][:,subs_idx[3]], color=c3, ax=ax[1][3], label = "het prob")

    ## hierarchical
    az.plot_dist(trace_hi['b'][:,subs_idx[0]], color=c4, ax=ax[1][0])
    az.plot_dist(trace_hi['b'][:,subs_idx[1]], color=c4, ax=ax[1][1])
    az.plot_dist(trace_hi['b'][:,subs_idx[2]], color=c4, ax=ax[1][2])
    az.plot_dist(trace_hi['b'][:,subs_idx[3]], color=c4, ax=ax[1][3], label = "hi prob")

    ## regression lines
    hdi_prob = 0.97
    subjects = reactions.Subject.unique()
    groups = reactions.groupby('Subject')
    for name,group in groups:
        if str(name) in subs:
            idx = subs.index(str(name))        
            ## scatter plots for observations
            scatter, = ax[2][idx].plot(group.Days, group.Reaction, c = 'navy', marker = "o", linestyle = '', label = "observations")
            ## homogenous non-prob
            x,y = line_points(reactions.Days, slope, intercept)
            hdi_hom_np, = ax[2][idx].plot(x , y, c = c2, linestyle = '-', linewidth = 3, label = "hom non-prob")  
            ## homogenous prob
            hdi_hom = az.plot_hdi(reactions.Days.unique(), rt_hom_prob, color = c1, hdi_prob=hdi_prob, ax = ax[2][idx], plot_kwargs={'label':'hom prob','alpha':0.5}, fill_kwargs={'alpha': 0.1})
            ## heterogenous prob
            idx_subjects = list(reactions.Subject.unique()).index(name)
            rt_het_prob_ = np.array([posterior_predictive_het['rt'][:,[s*len(reactions.Days.unique())+day for s in range(len(reactions.Subject.unique())) if s == idx_subjects]].flatten() for day in reactions.Days.unique()])
            rt_het_prob = np.array([[rt_het_prob_[:,draw] for draw in range(rt_het_prob_.shape[1])]])
            hdi_het = az.plot_hdi(reactions.Days.unique(), rt_het_prob, color = c3, hdi_prob=hdi_prob, ax = ax[2][idx], plot_kwargs={'label':'het prob','alpha':0.5}, fill_kwargs={'alpha': 0.1})
            ## hierarchical prob
            rt_hi_prob_ = np.array([posterior_predictive_hi['rt'][:,[s*len(reactions.Days.unique())+day for s in range(len(reactions.Subject.unique())) if s == idx_subjects]].flatten() for day in reactions.Days.unique()])
            rt_hi_prob = np.array([[rt_hi_prob_[:,draw] for draw in range(rt_hi_prob_.shape[1])]])
            hdi_hi = az.plot_hdi(reactions.Days.unique(), rt_hi_prob, color = c4, hdi_prob=hdi_prob, ax = ax[2][idx], plot_kwargs={'label':'hi prob','alpha':0.5}, fill_kwargs={'alpha': 0.1},backend_kwargs = {})

    ##
    for i,r in enumerate(ax):
        for j,c in enumerate(r):
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            if i<2:
                ax[i][j].set_ylabel("Probability Density")
            else:
                ax[i][j].set_ylabel("Reaction Time (ms)")
            if i == 0:
                ax[i][j].set_xlabel("a")
                ax[i][j].set_title("Driver "+subs[j])
                if j==3:
                    handles, labels = ax[i][j].get_legend_handles_labels()
                    ax[i][j].legend(handles, labels, loc='center right', bbox_to_anchor=(1.7, 0.5))
            elif i == 1:
                ax[i][j].set_xlabel("b")
                if j==3:
                    handles, labels = ax[i][j].get_legend_handles_labels()
                    ax[i][j].legend(handles, labels, loc='center right', bbox_to_anchor=(1.7, 0.5))
            elif i == 2:
                ax[i][j].set_xlabel("Days of Driving")
                ax[i][j].set_ylim(100, 550)
                ax[i][j].set_xlim(-1, 10)
                ax[i][j].legend()
                if j==3:
                    handles, labels = ax[i][j].get_legend_handles_labels()
                    ax[i][j].legend([handles[i] for i in [0,1,3,5,7]], [labels[i] for i in [0,1,3,5,7]],loc='center right', bbox_to_anchor=(1.7, 0.5))
                else:
                    ax[i][j].get_legend().remove()

    plt.savefig('ch1_Fig3.pdf', dpi=300, bbox_inches="tight", transparent=True)
    
def kruschke_style_posteriors(trace):
    ax = az.plot_posterior(trace, hdi_prob=0.97, textsize=20)
    fig = ax.ravel()[0].figure
    fig.savefig('reaction_posterior_arviz.pdf', dpi=300, bbox_inches="tight", transparent=True)