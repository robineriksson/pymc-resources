## This is a script version of the code in Chapter 2
##
## R. Marin 05/09/2022

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


ways = np.array([0, 3, 8, 9, 0])
print(ways/ways.sum())

x=stats.binom.pmf(6, n=9, p=0.5)
print(x)

def grid_approx(len=20):
    """
    Compute the posterior by the grid approximation of length len
    """

    # define grid
    p_grid = np.linspace(0,1,num=len)

    # define prior
    #prior = np.repeat(1,len) # type 1
    #prior[p_grid < 0.5] = 0  # type 2
    prior = np.exp(-5*np.abs(p_grid - 0.5))  # type 3
    prior = prior/prior.sum()

    # compute the likelihood at each value of the grid
    likelihood = stats.binom.pmf(6,n=9,p=p_grid)

    # compute the product of the likelihood and prior (-> posterior)
    posterior_unnorm = likelihood * prior

    # normalize the posterior
    posterior = posterior_unnorm / posterior_unnorm.sum()

    plt.plot(p_grid,posterior,'.-', label='posterior')
    plt.plot(p_grid,prior,'.-', label='prior')
    plt.legend()

    plt.show()

def quad_approx():
    data = np.repeat((0,1),(3,6))

    with pm.Model() as normal_approximation:
        p = pm.Uniform("p",0,1) # uniform prior
        w = pm.Binomial("w",n=len(data),p=p,observed=data.sum()) # binomial likelihood
        mean_q = pm.find_MAP()
        std_q = ( (1 / pm.find_hessian(mean_q, vars=[p])) ** 0.5)[0]

    # display summary of quadratic approximation
    print("  Mean, Standard deviation\np {:.2}, {:.2}".format(mean_q["p"], std_q[0]))

    return([mean_q, std_q])

x = quad_approx()

def CrI(mean_q,std_q, prob=0.89):
    norm = stats.norm(mean_q,std_q)
    z = stats.norm.ppf([ (1-prob) / 2, (1+prob)/2])
    pi = mean_q["p"] + std_q*z
    print(f'5.5%, 94.5% \n{pi[0]:.2}, {pi[1]:.2}')

CrI(x[0],x[1])

def true_vs_quad(mean_q, std_q):
    W, L = 6, 3
    x = np.linspace(0,1,100)
    plt.plot(x, stats.beta.pdf(x, W+1,L+1), label = 'true')
    plt.plot(x, stats.norm.pdf(x, mean_q["p"], std_q), label='quad')
    plt.legend()
    plt.show()

true_vs_quad(x[0],x[1])

def metro_approx(len=1_000):
    p = np.repeat(np.nan, len)
    p[1] = 0.5
    W,L = 6, 3
    for i in range(2,len):
        p_new = stats.norm(p[i-1], 0.1).rvs()
        p_new = np.abs(p_new) if p_new < 0 else p_new
        p_new = 2 - p_new if p_new > 1 else p_new
        q0 = stats.binom.pmf(W, W+L, p[i-1])
        q1 = stats.binom.pmf(W, W+L, p_new)
        p[i] = p_new if stats.uniform().rvs() < q1/q0 else p[i-1]

    x = np.linspace(0,1,100)
    az.plot_kde(p,label="mcmc approx")
    plt.plot(x, stats.beta.pdf(x, W+1,L+1), color='red',label = 'true')
    plt.legend()
    plt.show()

    return(p)

p = metro_approx()
