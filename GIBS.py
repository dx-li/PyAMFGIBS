# %%
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from scipy.spatial.distance import pdist
from pyminimax import minimax
from pyminimax import fcluster_prototype
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LassoCV
from tqdm import tqdm
# %%
def gibs_dist_metric(x, y):
    """
    Distance metric for GIBS clustering

    Parameters
    ----------
    x : 1 by n array
        First time series vector
    y : 1 by n array
        Second time series vector
    """
    return 1 - np.abs(np.corrcoef(x, y)[0][1])

def get_dist_matrix(data):
    """
    Compute distance matrix for GIBS clustering

    Parameters
    ----------
    data : n by m array
        Time series data
    """
    return pdist(data.T, metric=gibs_dist_metric)

def orthogonlize(data):
    """
    Orthogonalize time series data by projecting onto the first column using regression
    The first column should contain the market return minus the risk free rate

    Parameters
    ----------
    data : n by p array
        Time series data
    """
    column_names = data.columns
    data = data.to_numpy()
    #get first column
    first_col = data[:, 0]

    #get data without first column
    data = data[:, 1:]

    #get regression coefficients
    reg = lm.LinearRegression()
    reg.fit(first_col.reshape(-1, 1), data)

    #get orthogonalized data
    data = data - reg.predict(first_col.reshape(-1, 1))

    #add first column back to data
    data = np.hstack((first_col.reshape(-1, 1), data))

    return pd.DataFrame(data, columns = column_names)

def gibs_prototype_cluster(data):
    """
    Cluster time series data and returns prototypes, input orthogonalized data

    Parameters
    ----------
    data : n by p array
        Time series data
    """
    

    #get distance matrix
    dist = get_dist_matrix(data)

    #get minimax linkage
    Z = minimax(dist, return_prototype = True)

    #get cluster labels
    labels = fcluster_prototype(Z, t = 0.8, criterion = 'distance')

    #turn labels into a dataframe
    labels = pd.DataFrame(labels)
    #remove duplicates
    labels = labels.drop_duplicates()
    #get prototype indices
    prototypes = labels[1].to_numpy()

    return prototypes

def get_group_prototypes(data_orth, bath_size):
    """
    Find representative prototypes for each group of time series data

    Parameters
    ----------
    data : n by p array
        Time series data
    """
    data_orth = data_orth.iloc[:, 1:]
    #get number of groups
    num_groups = data_orth.shape[1] // bath_size

    #initialize array to hold representative prototypes
    rep_prototypes = []

    #loop through each group
    for i in range(num_groups):
        #get data for group
        group_data = data_orth.iloc[:, i * bath_size : (i + 1) * bath_size]
        #get representative prototype for group
        group_proto = list(group_data.iloc[:, gibs_prototype_cluster(group_data)])
        rep_prototypes += group_proto
    rep_prototypes.append('market_return')
    return rep_prototypes


def get_gibs_prototypes(data):
    #orthogonalize data
    data_orth = orthogonlize(data)
    #transform data into groups and find representative prototypes for each group
    rep_prototypes = get_group_prototypes(data_orth, 40)
    #print(rep_prototypes)
    #find prototypes of representative prototypes
    prototypes = gibs_prototype_cluster(data_orth[rep_prototypes])
    prototypes = list(data.iloc[:, prototypes].columns)
    return prototypes
    
def get_gibs_amf(stock, basis, prototypes):
    #run CV LASSO regression using prototypes
    lasso = LassoCV(cv = 5, max_iter = 10000, fit_intercept = True)
    lasso.fit(basis[prototypes], stock)
    #get indices of non-zero coefficients
    non_zero = np.where(lasso.coef_ != 0)[0]
    #run OLS regression using prototypes that have non-zero coefficients
    #add intercept
    ols = OLS(stock, add_constant(basis[prototypes].iloc[:, non_zero]))
    ols = ols.fit()
    return ols


# %%
if __name__ == '__main__':
    np.random.seed(10)
    #simulate data
    # Set the number of time points, the number of stocks, and the number of basis assets
    n = 300
    num_stocks = 5
    num_basis = 1000
    batch_size = 40

    # Set the MAR parameters
    rho = 0.8
    sigma = 0.1

    # Generate the stock returns using an MAR model
    returns = np.zeros((n, num_stocks))
    for i in range(1, n):
        returns[i] = rho * returns[i-1] + sigma * np.random.normal(size=(num_stocks))

    # Generate the basis assets in batches of 40, with each batch highly correlated
    basis = np.zeros((n, num_basis))
    for i in range(1, n):
        #pick random rho between 0.7 and 0.95
        rho = np.random.uniform(0.7, 0.95)
        for j in range(0, num_basis, batch_size):
            batch = rho * basis[i-1, j:j+batch_size] + sigma * np.random.normal(size=(batch_size))
            basis[i, j:j+batch_size] = batch

    # Generate the market return
    market_return = returns.mean(axis=1)

    # Create Pandas DataFrames for the stock returns, basis assets, and market return
    returns_df = pd.DataFrame(returns, columns=['stock' + str(i) for i in range(num_stocks)])
    basis_df = pd.DataFrame(basis, columns=['basis' + str(i) for i in range(num_basis)])
    market_return_df = pd.DataFrame(market_return, columns=['market_return'])

    # Concatenate market_return_df and basis_df into a single DataFrame
    data = pd.concat([market_return_df, basis_df], axis=1)

    prototypes = get_gibs_prototypes(data)

    print(f'There are {len(prototypes)} prototypes: {prototypes}')

    for stock in tqdm(returns_df.columns):
        amf = get_gibs_amf(returns_df[stock], data, prototypes)
        print(amf.summary())
        #plot fitted values and actual values with same scale
        plt.figure(figsize = (10, 5))
        plt.plot(amf.fittedvalues, label = 'fitted')
        plt.plot(returns_df[stock], label = 'actual')
        plt.legend()
        #add R^2 in title
        plt.title(f'{stock}, R^2 = ' + str(amf.rsquared)+ f", {len(amf.params) - 1} relevant prototypes)")
        plt.ylim(-0.5, 0.5)
        plt.show()
# %%
