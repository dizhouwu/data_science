#!/usr/bin/env python
# coding: utf-8

# reference: https://github.com/FranklinMa810/PortfolioManagement/blob/master/lab_23.ipynb
# 
# 
# https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf

# In[14]:


import numpy as np
import pandas as pd

def as_colvec(x):
    if x.ndim == 2:
        return x
    else:
        return np.expand_dims(x, axis=1)


# In[15]:


np.arange(4)


# In[16]:


as_colvec(np.arange(4))


# Recall that the first step in the Black Litterman procedure was to reverse engineer the implied returns vector $\pi$ from a set of portfolio weights $w$.
# $$\pi = \delta\Sigma w$$ 
# "This is performed by the following code:"

# The Master Formula
# The first step of the procedure is a reverse-optimization step that infers the implied returns vector $\pi$ that are implied by the equilibrium weights $w$ using the formula:
# 
# $$\pi = \delta\Sigma w$$
# Next, the posterior returns and covariances are obtained from the Black-Litterman Master Formula which is the following set of equations:
# 
# $$
# \label{eq:blMuOrig}
# \mu^{BL} = [(\tau\Sigma)^{-1} + P \Omega^{-1} P]^{-1}[(\tau\Sigma)^{-1} \pi + P \Omega^{-1} Q]
# $$$$
# \label{eq:blSigmaOrig}
# \Sigma^{BL} = \Sigma + [(\tau\Sigma)^{-1} + P \Omega^{-1} P]^{-1}
# $$
# Inverting $\Omega$
# While the master formulas identified in Equation \ref{eq:blMuOrig} and Equation \ref{eq:blSigmaOrig} are frequently easy to implement, they do involve the term $\Omega^{-1}$. Unfortuantely, $\Omega$ is sometimes non-invertible, which poses difficulties to implement the equations as-is. Fortunately the equations are easily transformed to a form that does not require this troublesome inversion. Therefore, frequently, implementations use the following equivalent versions of these equations which are sometimes computationally more stable, since they do not involve inverting $\Omega$. Derivations of these alternate forms are provided in the appendices of \cite{walters2011black}:
# 
# $$
# \label{eq:blMu}
# \mu^{BL} = \pi + \tau \Sigma P^T[(P \tau \Sigma P^T) + \Omega]^{-1}[Q - P \pi]
# $$$$
# \label{eq:blSigma}
# \Sigma^{BL} = \Sigma + \tau \Sigma - \tau\Sigma P^T(P \tau \Sigma P^T + \Omega)^{-1} P \tau \Sigma
# $$

# In[21]:


def implied_returns(delta, sigma, w):
    """

    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    ir = delta*sigma.dot(w).squeeze()# to get a series from a 1-column dataframe
    ir.name = 'Implied_Returns'
    return ir


# if the investor does not have a specific way to explicitly quantify the uncertaintly associated with the view in the $\Omega$ matrix, one could make the simplifying assumption that $\Omega$ is proportional to the variance of the prior.
# $$\Omega = diag(P (\tau \Sigma) P^T) $$

# In[20]:


# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    Q: the View Vector (k x 1 column vector)
    """
    helit_omega = p.dot(tau*sigma).dot(p.T)
    # make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)


# In[22]:


from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=0.02):
    """
        # Computes the posterior expected returns based on 
        # the original black litterman reference model
        #
        # W.prior must be an N x 1 vector of weights, a Series
        # Sigma.prior is an N x N covariance matrix, a DataFrame
        # P must be a K x N matrix linking Q and the Assets, a DataFrame
        # Q must be an K x 1 vector of views, a Series
        # Omega must be a K x K matrix a DataFrame, or None
        # if Omega is None, we assume it is
        #    proportional to variance of the prior
        # delta and tau are scalars
    """
    if not omega:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    N = w_prior.shape[0] # number of assets
    K = q.shape[0] # number of views
    # Step1, reverse engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior, w_prior)
    # adjust (scale) sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T)+omega).dot(q-p.dot(pi).values))
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


# 
# A Simple Example: Absolute Views
# We start with a simple 2-Asset example. Let's start with an example from Statistical Models and Methods for Financial Markets (Springer Texts in Statistics) 2008th Edition, Tze Lai and Haipeng Xing.
# 
# Consider the portfolio consisting of just two stocks: Intel (INTC) and Pfizer (PFE).
# 
# From Table 3.1 on page 72 of the book, we obtain the covariance matrix (multipled by $10^4$)
# 
# $$\begin{array}{lcc}
# INTC &amp; 46.0 &amp; 1.06 \\
# PFE   &amp; 1.06 &amp; 5.33
# \end{array}$$
# Assume that Intel has a market capitalization of approximately USD 80B and that of Pfizer is approximately USD 100B (this is not quite accurate, but works just fine as an example!). Thus, if you held a market-cap weighted portfolio you would hold INTC and PFE with the following weights: $W_{INTC} = 80/180 = 44\%, W_{PFE} = 100/180 = 56\%$. These appear to be reasonable weights without an extreme allocation to either stock, even though Pfizer is slightly overweighted.
# 
# We can compute the equilibrium implied returns $\pi$ as follows:

# In[23]:


tickers = ['INTC', 'PFE']
s = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) *  10E-4
pi = implied_returns(delta=2.5, sigma=s, w=pd.Series([.44, .56], index=tickers))
pi


# 
# Thus the equilibrium implied returns for INTC are a bit more than 5\% and a bit less than 1\% for PFE.
# 
# Assume that the investor thinks that Intel will return 2\% and that Pfizer is poised to rebounce, and will return 4\% . We can now examine the optimal weights according to the Markowitz procedure. What would happen if we used these expected returns to compute the Optimal Max Sharpe Ratio portfolio?
# 
# The Max Sharpe Ratio (MSR) Portfolio weights are easily computed in explicit form if there are no constraints on the weights. The weights are given by the expression (e.g. See \cite{campbell1996econometrics} page 188 Equation 5.2.28):
# 
# $$ W_{MSR} = \frac{\Sigma^{-1}\mu_e}{\bf{1}^T \Sigma^{-1}\mu_e} $$

# In[26]:


def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)
def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w


# In[27]:


mu_exp = pd.Series([.02, .04],index=tickers) # INTC and PFE
np.round(w_msr(s, mu_exp)*100, 2)


# In[28]:


# Absolute view 1: INTC will return 2%
# Absolute view 2: PFE will return 4%
q = pd.Series({'INTC': 0.02, 'PFE': 0.04})

# The Pick Matrix
# For View 2, it is for PFE
p = pd.DataFrame([
# For View 1, this is for INTC
    {'INTC': 1, 'PFE': 0},
# For View 2, it is for PFE
    {'INTC': 0, 'PFE': 1}
    ])

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_prior=pd.Series({'INTC':.44, 'PFE':.56}), sigma_prior=s, p=p, q=q)
# Black Litterman Implied Mu
bl_mu


# In[29]:


# Use the Black Litterman expected returns to get the Optimal Markowitz weights
w_msr(bl_sigma, bl_mu)


# In[30]:


pi


# A Simple Example: Relative Views
# In this example, we examine relative views. We stick with our simple 2-stock example. Recall that the Cap-Weighted implied expected returns are:

# In[32]:


pi


# In[33]:


q = pd.Series([
# Relative View 1: INTC will outperform PFE by 2%
  0.02
    ]
)
# The Pick Matrix
p = pd.DataFrame([
  # For View 1, this is for INTC outperforming PFE
  {'INTC': +1, 'PFE': -1}
])

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_prior=pd.Series({'INTC': .44, 'PFE': .56}), sigma_prior=s, p=p, q=q)
# Black Litterman Implied Mu
bl_mu


# In[34]:


pi[0]-pi[1]


# In[35]:


bl_mu[0]-bl_mu[1]


# In[36]:


# Use the Black Litterman expected returns and covariance matrix
w_msr(bl_sigma, bl_mu)


# In[37]:


w_msr(s, [.03, .01])


# In[38]:


w_msr(s, [.02, .0])


# Reproducing the He-Litterman (1999) Results
# We now reproduce the results in the He-Litterman paper that first detailed the steps in the procedure. We obtained the data by typing it in from the He-Litterman tables, and used it to test the implementation.
# 
# The He-Litterman example involves an international allocation between 7 countries. The data is as follows:

# In[39]:



# The 7 countries ...
countries  = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US'] 
# Table 1 of the He-Litterman paper
# Correlation Matrix
rho = pd.DataFrame([
    [1.000,0.488,0.478,0.515,0.439,0.512,0.491],
    [0.488,1.000,0.664,0.655,0.310,0.608,0.779],
    [0.478,0.664,1.000,0.861,0.355,0.783,0.668],
    [0.515,0.655,0.861,1.000,0.354,0.777,0.653],
    [0.439,0.310,0.355,0.354,1.000,0.405,0.306],
    [0.512,0.608,0.783,0.777,0.405,1.000,0.652],
    [0.491,0.779,0.668,0.653,0.306,0.652,1.000]
], index=countries, columns=countries)

# Table 2 of the He-Litterman paper: volatilities
vols = pd.DataFrame([0.160,0.203,0.248,0.271,0.210,0.200,0.187],index=countries, columns=["vol"]) 
# Table 2 of the He-Litterman paper: cap-weights
w_eq = pd.DataFrame([0.016,0.022,0.052,0.055,0.116,0.124,0.615], index=countries, columns=["CapWeight"])
# Compute the Covariance Matrix
sigma_prior = vols.dot(vols.T) * rho
# Compute Pi and compare:
pi = implied_returns(delta=2.5, sigma=sigma_prior, w=w_eq)
(pi*100).round(1)


# Next, we impose the view that German equities will outperform the rest of European equities by 5\%.
# 
# The other European equities are France and the UK. We split the outperformance proportional to the Market Caps of France and the UK.

# In[40]:


# Germany will outperform other European Equities (i.e. FR and UK) by 5%
q = pd.Series([.05]) # just one view
# start with a single view, all zeros and overwrite the specific view
p = pd.DataFrame([0.]*len(countries), index=countries).T
# find the relative market caps of FR and UK to split the
# relative outperformance of DE ...
w_fr =  w_eq.loc["FR"]/(w_eq.loc["FR"]+w_eq.loc["UK"])
w_uk =  w_eq.loc["UK"]/(w_eq.loc["FR"]+w_eq.loc["UK"])
p.iloc[0]['DE'] = 1.
p.iloc[0]['FR'] = -w_fr
p.iloc[0]['UK'] = -w_uk
(p*100).round(1)


# In[41]:


delta = 2.5
tau = 0.05 # from Footnote 8
# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau = tau)
(bl_mu*100).round(1)


# In[42]:


def w_star(delta, sigma, mu):
    return (inverse(sigma).dot(mu))/delta

wstar = w_star(delta=2.5, sigma=bl_sigma, mu=bl_mu)
# display w*
(wstar*100).round(1)


# View 4: Increasing View Uncertainty
# As a final step, He and Litterman demonstrate the effect of $\Omega$. They increase the uncertainty associated with the first of the two views (i.e. the one that Germany will outperform the rest of Europe). First we compute the default value of $\Omega$ and then increase the uncertainty associated with the first view alone.

# In[44]:


# This is the default "Proportional to Prior" assumption
omega = proportional_prior(sigma_prior, tau, p)
# Now, double the uncertainty associated with View 1
omega.iloc[0,0] = 2*omega.iloc[0,0]
np.round(p.T*100, 1)


# In[ ]:




