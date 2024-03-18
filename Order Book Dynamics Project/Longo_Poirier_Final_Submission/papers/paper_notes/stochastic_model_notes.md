# Stochastic Model Notes

* Models LBO as a continuous-time Markov process, and the model balances between:
  * Calibration to high-frequency data
  * Reproduction of empirical order book features
  * Being analytically tractible (explicit/closed solutions)
* Three conditional probabilities to be calculated using a Laplace transform:
  * Mid-price increase
  * Executing a bid before the ask quote moves
  * Executing a buy and sell order at best quotes before price moves
* More simple model than all-encompassing statistics/game theory model, but produces parameters that can be estimated efficiently
* Consider $X(t) = (X_i(t))_{i\in [n]}$. Here, price is multiples of tick price. Then at time $t$, $X_i(t)<0$ represents $|X_i(t)|$ quantity of bid order at price $i$ and $X_i(t)>0$ represents $|X_i(t)|$ quantity of ask order at price $i$.
* Consider four metrics:
  * $p_A(t)$ as the lowest ask price for $X_p(t)>0$ with $p=1,...,n$.
  * $p_B(t)$ as the highest bid price for $X_p(t)<0$ with $p=1,...,n$.
  * $Q_i^B(t)$ as the number of shares of a bid price that is $i$ away from the best ask or 0 if $i \geq p_A(t)$.
  * $Q_i^A(t)$ as the number of shares of an ask price that is $i$ away from the best bid or 0 if $i \geq n-p_B(t)$.
* Note that $X(t)$ and $(p_A(t), p_B(t), Q^B(t), Q^A(t))$ contain the same information, but the later shows the shape/depth of the book relative to the current best quotes.
* Order book flows are modeled by a Poisson process, with the following assumptions:
  * Limit buy orders arrive at distance $i$ from opposite best quote at independent, exponential times with rate $\lambda(i)$.
  * Market buy orders arrive with independent, exponential rate $\mu$.
  * Cancellation orders at a distance $i$ from the opposite best quote arrive at a proportional (to the number of outstanding orders) rate of $\theta(i)\cdot x$, where $x$ is the level.
  * All three above are mutually independent.
* You can model limit order, market order, and cancellation transitions using a continuous-time Markov chain
* Overall, the paper derives market book transitions (i.e. probabilities) conditioned on the state spaces imposed by the model assumptions. The empirical results are not impressive, but could be useful for thinking of different features to implement in our own model.
