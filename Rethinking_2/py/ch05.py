import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn as sns
from theano import shared

d = pd.read_csv("../Data/WaffleDivorce.csv", delimiter=";")

def standardize(series):
    series_std = (series - series.mean()) / series.std()
    return series_std

d["Divorce_std"] = standardize(d.Divorce)
d["Marriage_std"] = standardize(d.Marriage)
d["MedianAgeMarriage_std"] = standardize(d.MedianAgeMarriage)

def multi_M(data=d):
    with pm.Model() as mreg_divorce:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bm = pm.Normal("bm", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma", 1)
        mu = pm.Deterministic("mu", a + bm*d.Marriage_std)

        divorce = pm.Normal("divorce", mu=mu, sigma=sigma, observed=d.Divorce_std)
        trace = pm.sample(1000)
    return trace, mreg_divorce

def multi_A(data=d):
    with pm.Model() as mreg_divorce:
        a = pm.Normal("a", mu=0, sigma=0.2)
        ba = pm.Normal("ba", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma", 1)
        mu = pm.Deterministic("mu", a + ba*d.MedianAgeMarriage_std)

        divorce = pm.Normal("divorce", mu=mu, sigma=sigma, observed=d.Divorce_std)
        trace = pm.sample(1000)
    return trace, mreg_divorce

def multi_MA(data=d):
    with pm.Model() as mreg_divorce:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bm = pm.Normal("bm", mu=0, sigma=0.5)
        ba = pm.Normal("ba", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma", 1)
        mu = pm.Deterministic("mu", a + bm*d.Marriage_std + ba*d.MedianAgeMarriage_std)

        divorce = pm.Normal("divorce", mu=mu, sigma=sigma, observed=d.Divorce_std)
        trace = pm.sample(1000)
    return trace, mreg_divorce

yM = multi_M(d)
yA = multi_A(d)
yMA = multi_MA(d)
var_names = ["a", "bm", "ba", "sigma"]
#az.plot_trace(y[0],var_names=var_names); plt.show()
#az.summary(y[0],var_names=var_names)

az.plot_forest([yM[0], yA[0], yMA[0]],
               model_names=["M", "A", "MA"],
               var_names=var_names, combined=True); plt.show()

def pred_res_plot(data):
    with pm.Model() as pred_res:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bAM = pm.Normal("bAM", mu=0, sigma=0.2)
        sigma = pm.Exponential("sigma",1)

        mu = pm.Deterministic("mu", a + bAM*data["MedianAgeMarriage_std"].values)

        marriage = pm.Normal("marriage", mu=mu, sigma=sigma, observed=data["Marriage_std"].values)
        prior_samples = pm.sample_prior_predictive()
        trace = pm.sample(1000)
    return trace, pred_res, prior_samples

y = pred_res_plot(d)

def plot_res(data, y):
    mu_mean = y[0]["mu"].mean(axis=0)
    residuals = data["Marriage_std"] - mu_mean
    plt.plot(residuals)
    plt.show()

plot_res(d,y)


with yMA[1]:
    xx=pm.sample_posterior_predictive(yMA[0],var_names=["mu","divorce"],samples=100)

def plot_postpred(data=d,xx=xx):
    mu_mean = xx["mu"].mean(axis=0)
    mu_hdi = az.hdi(xx["mu"], 0.89)

    D_sim = xx["divorce"].mean(axis=0)
    D_hdi = az.hdi(xx["divorce"], 0.89)

    fig, ax = plt.subplots()
    plt.errorbar(data["Divorce_std"].values,
                  D_sim,
                  yerr = np.abs(D_sim - mu_hdi.T),
                  fmt="C0o")
    ax.scatter(data["Divorce_std"].values, D_sim)

    min_x, max_x = data["Divorce_std"].min(), data["Divorce_std"].max()
    ax.plot([min_x, max_x], [min_x, max_x], '--k')
    ax.set_ylabel('Predicted divorce')
    ax.set_xlabel('Observed divorce')
    plt.show()

plot_postpred(d,xx)


def spurr(N=100):
    x_real = stats.norm.rvs(size=N)
    x_spur = stats.norm.rvs(loc=x_real, size=N)
    y = stats.norm.rvs(loc=x_real, size=N)
    d = pd.DataFrame({"y":y, "x_real":x_real, "x_spur":x_spur})
    sns.pairplot(d)
    plt.show()

spurr(100)


marriage_shared = shared(d["Marriage_std"].values)
age_shared = shared(d["MedianAgeMarriage_std"].values)

def count_fac(data=d, m_s=marriage_shared, a_s=age_shared):
    with pm.Model() as m5_3_A:
        ## A --> D <-- M
        a = pm.Normal("a", mu=0, sigma=0.2)
        bM = pm.Normal("bM", mu=0, sigma=0.5)
        bA = pm.Normal("bA", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", a + bM*m_s + bA*a_s)
        divorce = pm.Normal("divorce", mu=mu, sigma=sigma, observed=data["Divorce_std"].values)
        ## A --> M
        aM = pm.Normal("aM", mu=0, sigma=0.2)
        bAM = pm.Normal("bAM", mu=0, sigma=0.5)
        sigma_M = pm.Exponential("sigma_M", 1)
        mu_M = pm.Deterministic("mu_M", aM + bAM*a_s)
        marriage = pm.Normal("marriage", mu=mu_M, sigma=sigma_M, observed=data["Marriage_std"].values)

        trace = pm.sample(1000)
    return trace, m5_3_A

y = count_fac()

A_seq = np.linspace(-2,2,50)
age_shared.set_value(A_seq)
with y[1]:
    m5_3_M_marriage = pm.sample_posterior_predictive(y[0])

def count_fac_plot(A_seq=A_seq,post=m5_3_M_marriage):
    fig, ax = plt.subplots(1,2)
    az.plot_hdi(A_seq, post["divorce"],0.89, ax=ax[0])
    ax[0].plot(A_seq, post["divorce"].mean(axis=0))
    ax[0].set_title('Total counterfactual effect of A on D')

    az.plot_hdi(A_seq, post["marriage"],0.89, ax=ax[1])
    ax[1].plot(A_seq, post["marriage"].mean(axis=0))
    ax[1].set_title('Total counterfactual effect of A on M')

    plt.show()

count_fac_plot()

def example_count(y):
    A_seq = (np.linspace(20, 30, 50)  - 26.1) / 1.24 # match the output size
    age_shared.set_value(A_seq)

    with y[1]:
        m5_3_M_ppc = pm.sample_posterior_predictive(y[0])

    # the difference between the first and last
    eff = m5_3_M_ppc["divorce"][:, -1].mean() - m5_3_M_ppc["divorce"][:, 0].mean()
    print(f'The average effect: {eff}')

example_count(y)


M_seq = np.linspace(-2,2,50)
marriage_shared.set_value(M_seq)
age_shared.set_value(np.zeros(50))

with y[1]:
    m5_3_M_ppc = pm.sample_posterior_predictive(y[0])

def count_fac_plot2(ppc=m5_3_M_ppc):
    fig, ax = plt.subplots()
    az.plot_hdi(A_seq, ppc["divorce"],0.89, ax=ax)
    ax.plot(A_seq, ppc["divorce"].mean(axis=0))
    ax.set_title('Total counterfactual effect of M on D')

    plt.show()

count_fac_plot2(m5_3_M_ppc)

d = pd.read_csv("../Data/milk.csv", sep=";")
d["K"] = standardize(d["kcal.per.g"])
d["N"] = standardize(d["neocortex.perc"])
d["M"] = standardize(d["mass"])


shared_N = shared(dcc["N"].values)

def monkey1(data=d, s_N=shared_N):
    with pm.Model() as m5_5_draft:
        a = pm.Normal("a", mu=0, sigma=1)
        bN = pm.Normal("bN", mu=0, sigma=1)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", a + bN*s_N)
        K = pm.Normal("K", mu=mu, sigma=sigma, observed=data.K)
        trace = pm.sample(1000)
    return trace, m5_5_draft

#y = monkey1(d)

dcc = d.dropna(subset="N")

y = monkey1(dcc, shared_N)

def monkey_prior(y):
    xseq = np.array([-2, 2])
    shared_N.set_value(xseq)
    with y[1]:
        pp = pm.sample_prior_predictive()

    fig,ax = plt.subplots()
    for i in range(50):
        ax.plot(xseq, pp['K'][i],'k',alpha=0.3)
    ax.set_xlabel('neocortex perc. (std)')
    ax.set_ylabel('kcal per g. (std)')
    ax.set_xlim(xseq)
    ax.set_ylim(xseq)
    plt.show()
    return(pp)

pp = monkey_prior(y)

def monkey2(data=d, s_N=shared_N):
    with pm.Model() as m5_5:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bN = pm.Normal("bN", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", a + bN*s_N)
        K = pm.Normal("K", mu=mu, sigma=sigma, observed=data.K)
        trace = pm.sample(1000)
    return trace, m5_5

shared_N = shared(dcc["N"].values)
y = monkey2(dcc, shared_N)

pp = monkey_prior(y)

trace=az.from_pymc3(y[0])
az.summary(trace, var_names=["a","bN","sigma"])

def monkey_postwplot(y=y,data=dcc):
    xseq = np.linspace(data["N"].min()-0.15, data["N"].max()+0.15, 30)
    shared_N.set_value(xseq)

    with y[1]:
        ppc = pm.sample_posterior_predictive(y[0], var_names=["mu"], samples=4_000)

    mu_mean = ppc["mu"].mean(axis=0)

    fig,ax = plt.subplots()
    az.plot_hdi(xseq, ppc["mu"], ax=ax)
    ax.plot(xseq, mu_mean)
    ax.scatter(data["N"], data["K"], facecolor="none", edgecolor='b')
    ax.set_xlabel('neocortex (std)')
    ax.set_ylabel('kcal (std)')
    plt.show()

monkey_postwplot(y,dcc)


shared_M = shared(dcc["M"].values)
def monkey2_M(data=d, s_M=shared_M):
    with pm.Model() as m5_5_M:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bM = pm.Normal("bM", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", a + bM*s_M)
        K = pm.Normal("K", mu=mu, sigma=sigma, observed=data.K)
        trace = pm.sample(1000)
    return trace, m5_5_M

yM = monkey2_M(dcc, shared_M)
trace_M=az.from_pymc3(yM[0])
az.summary(trace_M, var_names=["a","bM","sigma"])

def monkey2_NM(data=d, s_N=shared_N, s_M=shared_M):
    with pm.Model() as m5_5_M:
        a = pm.Normal("a", mu=0, sigma=0.2)
        bN = pm.Normal("bN", mu=0, sigma=0.5)
        bM = pm.Normal("bM", mu=0, sigma=0.5)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", a + bN*s_N + bM*s_M)
        K = pm.Normal("K", mu=mu, sigma=sigma, observed=data.K)
        trace = pm.sample(1000)
    return trace, m5_5_M

shared_N.set_value(dcc["N"])
yNM = monkey2_NM(dcc, shared_N, shared_M)
trace_NM=az.from_pymc3(yNM[0])
az.summary(trace_NM, var_names=["a", "bN", "bM", "sigma"])

def monkey_tree():
    az.plot_forest([trace, trace_M, trace_NM],
                   model_names=["N","M","NM"],
                   var_names=["bM","bN"],
                   combined=True)
    plt.show()

def monkey_counter(data=dcc, yNM=yNM):
    shared_N.set_value(np.zeros(30))
    seqx = np.linspace(data["M"].min() - 0.15, data["M"].max() + 0.15, 30)
    shared_M.set_value(seqx)

    with yNM[1]:
        ppc = pm.sample_posterior_predictive(yNM[0],var_names=["mu"],samples=4000)

    mu_mean = ppc["mu"].mean(axis=0)

    _,ax = plt.subplots()
    az.plot_hdi(seqx, ppc["mu"], ax=ax)
    ax.plot(seqx,mu_mean,'k')

    plt.show()

monkey_counter(dcc,yNM)

d = pd.read_csv("../Data/Howell1.csv", sep=";")

def height_sex1(data=d):
    with pm.Model() as m5_8:
        sigma = pm.Uniform("sigma", lower=0, upper=50)
        mu = pm.Normal("mu", mu=178, sigma=20, shape=2)
        height = pm.Normal("height", mu=mu[data.male.values], sigma=sigma, observed=data["height"])
        diff_fm = pm.Deterministic("diff_fm",mu[0] - mu[1])
        trace = pm.sample()
    return trace, m5_8

y=height_sex1(d)

d = pd.read_csv("../Data/milk.csv", sep=";")
d["clade_id"] = pd.Categorical(d["clade"]).codes
d["K"] = standardize(d["kcal.per.g"])

def milk_cat(data=d):
    with pm.Model() as m5_9:
        a = pm.Normal("a", mu=0, sigma=0.5, shape=data["clade_id"].max()+1)
        sigma = pm.Exponential("sigma",1)
        K = pm.Normal("K", mu=a[data["clade_id"]], sigma=sigma, observed=data["K"])
        trace = pm.sample(1000)
    return trace, m5_9

y = milk_cat(d)

az.plot_forest(y[0], var_names=["a"], combined=True)
plt.show()

def milk_cat2(data=d):
    house = np.random.randint(0, 4, size=d.shape[0])
    with pm.Model() as m5_10:
        mu_clade = pm.Normal("mu_clade", mu=0, sigma=0.5, shape=data["clade_id"].max()+1)
        mu_house = pm.Normal("mu_house", mu=0, sigma=0.5, shape=house.max()+1)
        sigma = pm.Exponential("sigma",1)
        mu = pm.Deterministic("mu", mu_clade[data["clade_id"]] +  mu_house[house])
        K = pm.Normal("K", mu=mu, sigma=sigma, observed=data["K"])
        trace = pm.sample(1000)
    return trace, m5_10

yy = milk_cat2(d)


az.plot_forest(yy[0], var_names=["mu_clade", "mu_house"], combined=True)
plt.show()

az.summary(yy[0], var_names=["mu_clade", "mu_house"])
