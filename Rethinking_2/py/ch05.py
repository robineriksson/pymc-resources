import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats

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


with y[1]:
    xx=pm.sample_posterior_predictive(y[0],var_names=["mu","marriage"],samples=100)
