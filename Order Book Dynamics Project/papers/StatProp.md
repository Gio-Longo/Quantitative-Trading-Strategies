$\newcommand{\book}{\mathcal{B}}$
$\newcommand{\midprice}{\mathcal{M}}$
$\newcommand{\time}{\mathcal{T}}$
$\newcommand{\confidence}{\mathcal{C}}$

# Notes of: Statistical properties of stock order books: empirical results and models
Feb 6th 2008

### Notation:
$a(t)$: the ask price at time $t$

$b(t)$: the bid price at time $t$

$m(t)$: the mid price at time $t$, where $m(t) = [a(t)+b(t)]/2$

$g(t)$: the bid-ask spread at time $t$, where $g(t) = a(t) - b(t)$

$\Delta$: The distance between a limit order and the midprice $m(t)$


### PDF of limit order distance

$$
    P(\Delta) \propto \frac{\Delta^\mu_0}{(\Delta_1 + \Delta)^{1+\mu}},\quad\Delta\geq 1
$$

with $\mu\approx 0.6$ (we may have to verify this)

This equation is quite hard to understand, here's what I think the authors are trying to say:

$\Delta_0$: Some normalization constant

$\Delta_1$: Some other constant

$\Delta$: Change in price 

This is essentially saying that as $\Delta\rightarrow\infty$, $P(\Delta)\rightarrow 0$. Thought the distance of an order may get more and more unlikely we see that it's still not trivial, and in fact quite larger than one may expect. 

If we recycle some of the constants from the paper that inspired this model ([paper](https://arxiv.org/pdf/cond-mat/0206280.pdf)), we can set $\Delta_1\approx 7$. We set $\Delta_0 \approx P_0$, and keep $\mu \approx 0.5$. Using this approach we can perhaps generalize some "base" distribution for every limit order and think about their effect on the book as a whole.

For instance, we can try to find when $\Delta$ is negative which implies that the book would be moving in the opposite direction, or tightening. Here's a plot from the report shouwing this:

![Delta Plot](../imgs/StatProp_Delta_Plot.png "Delta Plot")

### Understanding the "shape" of the order book

Order flow is at its maximum around the current price, however it has higher likelihood to get filled and disappear. It's therefore not super clear what the average shape of the book will be. The authors find that the volume at the bid (or ask) should be proportional to a gamma distribution:

$$
    R(V)\propto V^{\gamma-1}\exp\left(-\frac{V}{V_0}\right)
$$

with,

$\gamma\approx 0.75$

This implies that the most probable volume at best bid or ask, is very small, however empirically we observe that this can in fact fluctute quite a lot.
