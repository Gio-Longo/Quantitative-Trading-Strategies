$\newcommand{\book}{\mathcal{B}}$
$\newcommand{\midprice}{\mathcal{M}}$
$\newcommand{\time}{\mathcal{T}}$
$\newcommand{\confidence}{\mathcal{C}}$

# Notes of: Avellaneda-Stoikov market making model


Let the market price be modelled by:

$$
    dS_t = \sigma dW_t
$$

The model only assumes that one order is being traded at a time. We assume that the time at which transactions occur is purely random. 

We denote by $(N_t^b)_t$ and $(N_t^a)_t$, the two point processes modeling the number of assets that have been respectively bought and sold. The inventory of the market maker is modelled by :

$$
    q_t = N_t^b - N_t^a
$$

The intensity of the process of $(N_t^b)_t$ and $(N_t^a)_t$ are denoted by $(\lambda_t^b)_t$ and $(\lambda_t^a)_t$. 

The main assumptions of the model are:

$$
\lambda_t^b = \Lambda^b (\delta^b_t)
$$

and

$$
\lambda_t^a = \Lambda^a (\delta^a_t)
$$

where

$$
\delta^b_t = S_t - S_t^b
$$

and 
$$
\delta^a_t = S_t^a - S_t
$$

and where $\Lambda^b$ and $\Lambda^a$ are two positive and nonincreasing functions.

They focused on the specific case where

$$
    \Lambda^b(\delta) = \Lambda^a(\delta) = Ae^{-\mathcal{k}\delta}
$$

In this setting, $A$ characterizes the liquidity of the asset, and $\mathcal{k}$ characterizes the price sensitivity of market participants.


The amount of cash on the mm's account is modelled by:

$$
    dX_t = S^a_tdN_t^a -  S^b_t dN_t^b
$$

The goal is to maximize the CARA utility criterion:

$$
    \mathbb{E}[-\exp(-\gamma(X_T + q_TS_T - \ell(q_T)))]
$$
