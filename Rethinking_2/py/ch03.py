import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

def grid_approx(len=1_000):
    p_grid = np.linspace(0,1,len)
    prob_p = np.repeat(1,len)
    prob_data = stats.binom.pmf(8, n=15, p=p_grid)
    posterior = prob_data * prob_p
    posterior = posterior / posterior.sum()
    return([p_grid, posterior])

x = grid_approx()

plt.plot(x[0],x[1])
plt.show()

len=10_000
samples = np.random.choice(x[0],len,p=x[1],replace=True)

_, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(samples, 'o', alpha=0.2)
az.plot_kde(samples, ax=ax2)
plt.show()


((samples > 0.5) & (samples < 0.75)).sum()/len

np.percentile(samples,[10, 90])

az.hdi(samples,hdi_prob=0.5)

def grid_approx2(len=1_000, len_samples=10_000):
    p_grid = np.linspace(0,1,len)
    prob_p = np.repeat(1,len)
    prob_data = stats.binom.pmf(3, n=3, p=p_grid)
    posterior = prob_data * prob_p
    posterior = posterior / posterior.sum()
    samples = np.random.choice(p_grid, len_samples, p=posterior, replace=True)
    return(samples)

samples = grid_approx2()

def ch3M3(len=10_000):
    W, N = 8, 15
    x = grid_approx()
    samples = np.random.choice(x[0],len,p=x[1],replace=True)

    samples= stats.binom.rvs(1,p=samples,size=(N,samples.shape[0]))

    pred = samples.sum(axis=0)

    print(f'The 90% HPDI: {az.hdi(pred,hdi_prob=0.9)}')

    plt.hist(pred)
    #az.plot_kde(pred)

    # az.plot_kde(pred)
    plt.axvline(x=W)
    plt.show()
    return(pred)

# plt.plot(pred)
# plt.show()

samples=ch3M3(10_000)
