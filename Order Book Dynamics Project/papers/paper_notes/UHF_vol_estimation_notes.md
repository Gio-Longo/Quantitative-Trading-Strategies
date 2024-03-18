# Ultra High Frequency Volatility Estimation Notes

* Given the log-price $Y$ in HFT data is assumed to be the efficient log-price $X$ plus noise $\epsilon$.
* We say that $dX_t=\mu_tdt+\sigma_tdW_t$, but set $\mu_t=0$ since drift is statistically irrelevant when observing such small time scales.
* To ensure the robustness of noise analysis, it is preferred to depart from any assumptions on $\epsilon$.
* The parametric case of $\sigma_t$ has been solved -> $\epsilon$ produces consistent and asymptotically normal estimators of the parameters
  * In this case, misspecification of the marginal distribution of $\epsilon$ has no adverse consequences
* In the nonparametric case of $\sigma_t$ as an unrestricted stochastic process, we are interested in:

$$
\langle X, X\rangle_t = \int_0^T \sigma_t^2dt.
$$
* The quadratic variance term can be used to hedge derivatives portfolio, forcast future integrated volatility, etc.
* Solution to estimating the volatility is the "Two Scales Realized Volatility", which evaluates the quadratic variation at two different frequencies, averages the results over the sample period, and takes a linear combination of the result at the two frequencies -> produced consistant and asymptotically unbiased estimator of quadratic variation.
  * One drawback: this idea does not account for serial depenedence, so adaptations will be made to create a new version of the TSRV
* The new approach is called "Multiple Scales Realized Volatility", which will be used to analyze the impact of serial dependence in the noise
* When using the TSRV, we split the sample of $n$ observations (in seconds) into subsamples of size $\tilde{n}$ (into minutes). This reduces the bias term from $2nE[\epsilon^2]$ to $2\tilde{n}E[\epsilon^2]$, and we can approximate

$$
E[\epsilon^2] = \frac{1}{2n} \sum_{i=1}^n (Y_{t_{i+1}}-Y_{t_i})^2 = [Y, Y]_T^{(all)}.
$$

* Additionally, we say that $[Y, Y]_T^{(avg)}$ is 

$$
\frac{1}{K} \cdot \sum_{k=1}^K \sum_{i=1}^{\tilde{n}-k} (Y_{t_{i+1}}-Y_{t_i})^2.
$$

* With these approximations, we say the bias adjusted estimator for our quadratic variation is

$$
\widehat{\langle X, X\rangle}_T^{(tsrv)} = [Y, Y]_T^{(avg)} - \frac{\tilde{n}}{n}[Y, Y]_T^{(all)}.
$$

* An adjustment can be made for small sample sizes by setting $\widehat{\langle X, X\rangle}_T^{(tsrv, adj)}= \left(1-\frac{\tilde{n}}{n}\right)\widehat{\langle X, X\rangle}_T^{(tsrv)}$.
* More time scale samplings can be used to improve convergance rate -> take weighted average of $[Y, Y]_T^{(avg)}$.
* We now focus on a new, robust TSRV estimator that departs from the iid noise assumption -> same equations for $Y$ and $X$
* Consider $(K_i)$ to be a collection of integers in $1,..., n$. We say that the term $[Y, Y]_T^{(K_i)}$ is given by

$$
\frac{1}{K_i} \sum_{j=0}^{n-K_i} (Y_{t_{j+K_i}}-Y_{t_j})^2.
$$

* The MSRV estimator $\widehat{\langle X, X\rangle}_T^{(msrv)}$ is then given by

$$
\sum_{i=1}^M a_i\cdot [Y, Y]_T^{(K_i)} + 2[Y, Y]_T^{(all)}.
$$

* Here, we define

$$
a_i = \frac{i}{M^2} h\left(\frac{i}{M}\right) - \frac{1}{2M^2}\frac{i}{m}h'\left(\frac{i}{M}\right).
$$

* For the above equation, $h$ is a continuously differentiable real-value function that satisfies:

$$
\int_0^1 xh(x)dx = 1 \text{ and } \int_0^1 h(x)dx=0.
$$

* Note that this estimate was based on trade data and used a full-day worth of training data at second time scales -> bring this to order book application and use microsecond levels
* How do we use this MSRV estimate? That is for us to decide, but it could be used to predict futurue volatility, portfolio hedging, etc. Note that the term is just integrated volatility, i.e. the higher the MSRV the higher our volatility estimate. If we use the MSRV to predict future volatility, then we can adjust sizing and leverage based on our forecasts.
