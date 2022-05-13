import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
#import seaborn as sns

from scipy.interpolate import griddata

def football_flips():
    steps = 16
    repetitions = 1_000
    x = np.zeros([steps, repetitions])
    for i in range(x.shape[1]):
        x[1:,i] = np.cumsum(np.random.uniform(-1,1,steps-1))

    plt.plot(range(1,17),x[:,:],'b',alpha=0.05)
    plt.plot(range(1,17),x[:,0],'k')
    plt.axvline(4,linestyle='--')
    plt.axvline(8,linestyle='--')
    plt.axvline(16,linestyle='--')
    plt.show()
football_flips()

def organ_growth():
    x = np.random.uniform(1,1.1, size=(12,10_000)).prod(axis=0)
    x_small = np.random.uniform(1,1.01, size=(12,10_000)).prod(axis=0)
    x_large = np.random.uniform(1,1.5, size=(12,10_000)).prod(axis=0)

    _, (ax1,ax2,ax3) = plt.subplots(1,3)
    az.plot_kde(x,ax=ax1)
    az.plot_kde(x_small,ax=ax2)
    az.plot_kde(np.log(x_large),ax=ax3)
    plt.show()

organ_growth()


d = pd.read_csv("../Data/Howell1.csv",sep=';',header=0)
az.summary(d.to_dict(orient='list'),kind="stats")
d2 = d.loc[d.age>=18,:]

x = np.linspace(100,250,100)
y = stats.norm.pdf(x,178, 20)
plt.plot(x,y)
plt.show()

x = np.linspace(-10,60,100)
y = stats.uniform.pdf(x,0,50)
plt.plot(x,y)
plt.show()



def priorpred(size=10_000):
    sample_mu = stats.norm.rvs(size=size, loc=178, scale=100)
    sample_sigma = stats.uniform.rvs(size=size, loc=0, scale=50)
    prior_h = stats.norm.rvs(size=size, loc=sample_mu, scale=sample_sigma)
    az.plot_kde(prior_h)
    plt.show()
    return(0)

priorpred()

def grid(size=100,data=d2):
    post = np.mgrid[150:160:0.05, 7:9:0.05].reshape(2, -1).T

    likelihood = [
        sum(stats.norm.logpdf(data.height, loc=post[:,0][i], scale=post[:,1][i]))
        for i in range(len(post))
    ]

    post_prod = (
        likelihood
        + stats.norm.logpdf(post[:,0], loc=178, scale=20)
        + stats.uniform.logpdf(post[:,1], loc=0, scale=50)
    )

    post_prob = np.exp(post_prod - max(post_prod))
    return({"grid":post, "post":post_prob})

y = grid()

def gridplot(post, post_prob):
    xi = np.linspace(post[:,0].min(), post[:,0].max(), 100)
    yi = np.linspace(post[:,1].min(), post[:,1].max(), 100)
    zi = griddata((post[:,0], post[:,1]), post_prob, (xi[None,:], yi[:,None]))

    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.contour(xi,yi,zi)

    ax2.imshow(zi, origin="lower", extent=[150.0, 160.0, 7.0, 9.0], aspect="auto")
    plt.show()

gridplot(y["grid"], y["post"])

N = y["post"].size

sample_rows = np.random.choice(np.arange(N), size=100, p = y["post"], replace=True)


def samplepost(y,size=10_000):
    N = y["post"].size
    p = y["post"]/y["post"].sum()
    sample_rows = np.random.choice(np.arange(N), size=size, p = p, replace=True)
    sample_mu = y["grid"][sample_rows,0]
    sample_sigma = y["grid"][sample_rows,1]
    return(pd.DataFrame({"mu":sample_mu, "sigma":sample_sigma}))

sample = samplepost(y)


def quad(data=d2):
    with pm.Model() as quad_height:
        mu = pm.Normal("mu",mu=178,sd=20)
        sigma = pm.Uniform("sigma",lower=0,upper=50)
        height = pm.Normal("height",mu=mu,sd=sigma,observed=data.height)
        mean_mu = pm.find_MAP()
        std_mu = ((1/pm.find_hessian(mean_mu, vars=[mu,sigma]))**0.5)[0]
    return([mean_mu, std_mu])
y=quad(d2)

def traceit(data=d2,testval=True):
    if testval:
        with pm.Model() as quad_height:
            mu = pm.Normal("mu",mu=178,sd=20)
            sigma = pm.Uniform("sigma",lower=0,upper=50)
            height = pm.Normal("height",mu=mu,sd=sigma,observed=data.height)
            trace = pm.sample(1000,tune=1000)
    else:
        with pm.Model() as quad_height:
            mu = pm.Normal("mu",mu=178,sd=20, testval=data.height.mean())
            sigma = pm.Uniform("sigma",lower=0,upper=50, testval=data.heigth.sd())
            height = pm.Normal("height",mu=mu,sd=sigma,observed=data.height)
            trace = pm.sample(1000,tune=1000)
    return(trace)

#y0=traceit(d2,testval=False)
y1=traceit(d2,testval=True)

trace_df = pm.trace_to_dataframe(y1)

az.summary(y1,round_to=2,kind="stats")

plt.plot(d2.height, d2.weight, '.')
plt.show()


def priorlm(log=False):
    N = 100
    a = stats.norm.rvs(178,20,size=N).reshape(-1,1)
    if log:
        b = stats.lognorm.rvs(s=1,scale=1,size=N).reshape(-1,1)
    else:
        b = stats.norm.rvs(0,10,size=N).reshape(-1,1)
    x = np.linspace(d2.weight.min(), d2.weight.max(),100)
    plt.plot((a+b*(x - x.mean())).T,'k', alpha=0.1)
    plt.show()

#priorlm(False)
priorlm(True)

def traceit2(data=d2):
    xbar = data.weight.mean()
    with pm.Model() as lm:
            alpha = pm.Normal("alpha",mu=178,sd=20,testval=data.height.mean())
            beta = pm.LogNormal("beta",mu=1, sd=1)
            sigma = pm.Uniform("sigma",lower=0,upper=50,testval=data.height.std())
            mu = alpha + beta * (d2.weight - xbar)
            height = pm.Normal("height",mu=mu,sd=sigma,observed=data.height)
            trace = pm.sample(1000,tune=1000)
    return(trace)

y = traceit2(d2)
az.plot_trace(y)
plt.show()

az.summary(y,round_to=2,kind="stats")

trace_df = pm.trace_to_dataframe(y)

def lmplot(trace=trace_df, data=d2, N=10):

    # a_map = trace.alpha.mean()
    # b_map = trace.beta.mean()
    xbar = data.weight.mean()

    plt.plot(data.weight,data.height,"o",alpha=0.5)

    a = trace.alpha[:N].to_numpy().reshape(-1,1)
    b = trace.beta[:N].to_numpy().reshape(-1,1)
    x = data.weight.to_numpy()
    y=a+b*(x-xbar)


    plt.plot(data.weight,y.T,'k',alpha=0.1)
    plt.plot(data.weight,trace.alpha.mean() + trace.beta.mean() * (data.weight - xbar),'r')
    plt.show()
lmplot(N=100)

def probatx(x=50,trace=trace_df,data=d2):
    xbar = data.weight.mean()
    mu_at_x = trace.alpha.to_numpy().reshape(-1,1) \
    + trace.beta.to_numpy().reshape(-1,1) \
    * (50 - xbar)

    az.plot_kde(mu_at_x)
    plt.show()
    return(mu_at_x)
mux=probatx()


d = pd.read_csv("../Data/Howell1.csv",sep=';',header=0)

d2 = d.loc[d.age >= 18, :]

def traceit2(data=d2):
    xbar = data.weight.mean()
    with pm.Model() as lm:
        alpha = pm.Normal("alpha", mu=178, sigma=20)
        beta = pm.LogNormal("beta", mu=1,  sigma=1)
        sigma = pm.Uniform("sigma", lower=0, upper=50)
        mu = pm.Deterministic("mu",alpha + beta * (data.weight - xbar))
        height = pm.Normal("height", mu=mu, sd=sigma, observed=data.height)
        trace = pm.sample(1000, tune=1000)
    return(trace)

y = traceit2(d2)

def foo():
    N = [10, 50, 150, 352][0]
    dN = d2[:N]
    with pm.Model() as m_N:
        a = pm.Normal("a", mu=178, sigma=100)
        b = pm.Lognormal("b", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", lower=0, upper=50)
        mu = pm.Deterministic("mu", a + b * (dN.weight.to_numpy() - dN.weight.mean()))
        height = pm.Normal("height", mu=mu, sigma=sigma, observed=dN.height)
        trace_N = pm.sample(1000, tune=1000)
    return(trace_N)
foo()



def poly(data=d):
    data["weight_s"] = (data.weight - data.weight.mean())/data.weight.std()
    data["weight_s2"] = data.weight_s**2
    with pm.Model() as poly_height:
        a = pm.Normal("a", mu=178, sigma=20)
        b1 = pm.LogNormal("b1", mu=1, sigma=1)
        b2 = pm.Normal("b2", mu=0, sigma=1)
        mu = pm.Deterministic("mu", a + b1*d.weight_s.values + b2*d.weight_s2.values)
        sigma = pm.Uniform("s", lower=0, upper=50)
        height = pm.Normal("height", mu=mu, sigma=sigma, observed=data.height)
        trace = pm.sample(1_000, tune=1_000)
    return(trace)

y=poly(d)
