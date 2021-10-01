from datetime import date

import cvxpy as cp
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pypfopt as py
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

mylist = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "XOM",
          "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "NKE", "PFE",
          "PG", "RTX", "TRV", "UNH", "VZ",]

#   "V", "WBA", "WMT", "DIS"]

ESG_list = [75.51, 22.31, 13.68, 78.88, 73.85, 88.39, 3.12, 39.92, 20.00, 77.93,
            40.7, 3.15, 13.78, 8.24, 55.27, 52.24, 36.38, 31.08, 9.86, 39, 48.27,
            55, 18.04, 23.54, 16.69,]

#   11.41, 18.01, 49.23, 9]


def ESG_Portf(S, mylist, ESG_list, ESG_bound_U, ESG_bound_L):
    n = len(mylist)
    w = cp.Variable(n)

    Portfolio_ESG = w.T @ ESG_list
    objective = cp.Minimize(cp.quad_form(w, S))
    constraints = [cp.sum(w) == 1,
                   w >= 0.01,
                   Portfolio_ESG <= ESG_bound_U,
                   Portfolio_ESG >= ESG_bound_L, ]

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS, verbose=False, use_indirect=True)

    print("")
    print("optimal variables= ", w.value)
    print(sum(w.value))
    Portfolio_Rating = w.value @ ESG_list
    print(Portfolio_Rating)

    return w.value


def Mean_Variance(S, mylist):
    n = len(mylist)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, S))
    constraints = [cp.sum(w) == 1,
                   w >= 0.01, ]

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS, verbose=False, use_indirect=True)

    print("")
    print(sum(w.value))

    Portfolio_Rating = w.value @ ESG_list
    print(Portfolio_Rating)

    return w.value

ESG_arr = np.array(ESG_list)
print(ESG_arr)

start_date = "2000-03-11"
today = date.today().strftime("%Y-%m-%d")

df = pd.DataFrame()
n = 0

for stock in mylist:
    df[stock] = web.DataReader(stock, data_source="yahoo", start=start_date, end=today)['Adj Close']
print(df)

cov_sigma_semi = py.risk_models.exp_cov(df, returns_data=False, span=180, frequency=252, log_returns=False, )
Corr_Coefficient = py.risk_models.cov_to_corr(cov_sigma_semi)
Risk_model_PSD = py.risk_models.fix_nonpositive_semidefinite(Corr_Coefficient, fix_method='diag')

mu = expected_returns.mean_historical_return(df)
S = np.array(risk_models.CovarianceShrinkage(Risk_model_PSD).ledoit_wolf())

ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 1))

print(Mean_Variance(S, mylist))
print(ESG_Portf(S, mylist, ESG_arr, ESG_bound_U=21, ESG_bound_L=19))
print(ESG_Portf(S, mylist, ESG_arr, ESG_bound_U=41, ESG_bound_L=39))
print(ESG_Portf(S, mylist, ESG_arr, ESG_bound_U=61, ESG_bound_L=59))
