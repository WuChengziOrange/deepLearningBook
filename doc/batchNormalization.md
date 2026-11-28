# How to calculate the partial of batch normalization?

## 1. Normalization formula

$$
\mu = \frac{1}{m} \sum_{i=1}^m x_i
$$

$$
\sigma ^ 2 = Var(X)=  \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sigma}
$$

## 2. Normalizition data  attribution

So the $\hat{x}_i$ has the attribute below:

$$
\hat \mu = 0
$$

$$
\hat{\sigma} = 1
$$

## 3. Calculate partial formula

Lets calculate $\frac{\partial \hat x_i} {\partial x_i}$:

$$
\frac{\partial \hat x_i} {\partial x_i} = \frac{\partial \frac{x_i - \mu}{\sigma}}{\partial x_i}
$$

it looks very difficult to calculate this formula, so we need to expand this expression.

### 3.1 Basic element partial calculation

It looks complex, so we can calculate some basic element first:

$$
\frac{\partial x_i} {\partial x_i} = 1
$$

$$
\frac {\partial \mu} {\partial x_i} = \frac{1} {m} \tag{3.1.1}
$$

### 3.2 $\sigma$ partial calculation

here is a much complex element $\frac {\partial \sigma} {\partial x_i}$, since:

$$
\sigma =\sqrt{Var(X)}= \sqrt{\frac{1}{m}\sum_{i=1}^m (x_i-\mu)^2}
$$

so:

$$
\begin{align*}
\frac{\partial \sigma}{\partial x_i} &= \frac{1}{2} Var(X)^ {-\frac{1}{2}}
\frac{\partial Var(X)} {\partial x_i}\\
&=\frac{1}{2}({\sigma^2})^{-1/2} \frac{\partial Var(X)}{\partial x_i}\\
&=\frac{1}{2\sigma}\frac{\partial Var(X)}{\partial x_i} \tag{3.2.1}
\end{align*}
$$

thanks to the chain rule, we can continue calculate:

$$
\frac{\partial Var(X)}{\partial x_i} =
\frac
	{
  		\partial
  		\frac{1}{m}
      	\sum_{k=1}^m
      	(x_k - \mu)^2
	}
	{\partial x_i}
$$

now we need to case analysis:

$$
\frac{\partial Var(X)}{\partial x_i} = \sum
\begin{cases}
  \frac{1}{m} \frac{\partial (x_i - \mu)^2} {\partial x_i}
  =\frac{1}{m}\frac{\partial (x_i - \mu)^2}{\partial (x_i-\mu)}
  \frac{\partial (x_i-\mu)}{\partial x_i}
  =\frac{2(x_i-\mu)}{m}[\frac{\partial x_i}{\partial x_i} - \frac{\partial \mu}{\partial x_i}]
  	, & i=k \\

  \frac{1}{m} \frac{\partial \sum_{k=1}^m(x_k - \mu)^2}{\partial x_i}
  =\frac{1}{m}\sum_{k=1}^m \frac{\partial (x_k-\mu)^2}{\partial (x_k-\mu)}
  \frac{\partial (x_k-\mu)}{\partial x_i}
  = \frac{1}{m} \sum_{k=1}^m 2(x_k-\mu)(\frac{\partial x_k}{\partial x_i} - \frac{\partial \mu}{\partial x_i})
  	, & i\neq k \\
\end{cases}
$$

Rearranging gives:

$$
\begin{align*}
\frac{\partial Var(X)}{\partial x_i}
&=\frac{2(x_i-\mu)(1-1/m)}{m}
+\frac{2}{m}\sum_{k=1,k\neq i}^m (x_k-\mu)(0-1/m) \\
&=\frac{2}{m}[x_i-\mu + (x_i-\mu)(-1/m)+\sum_{k=1,k\neq i}^m (x_k-\mu)(-1/m)]\\
&=\frac{2}{m}[x_i-\mu + \sum_{k=1}^m (x_k-\mu)(-1/m)]\\
&=\frac{2}{m}[x_i-\mu - \frac{1}m(\sum_{k=1}^m x_k - m\mu)]\\
&=\frac{2}{m}(x_i-\mu -\frac{1}{m}\sum_{k=1}^m x_k + \mu)\\
&=\frac{2}{m}(x_i-\mu) \tag{3.2.2}
\end{align*}
$$

According to Eq. (3.2.1) and (3.2.2), so we got the $\sigma$ partial formula:

$$
\frac{\partial \sigma}{\partial x_i}
  =\frac{1}{2\sigma}\frac{2}{m}(x_i-\mu)
  =\frac{x_i-\mu}{m\sigma} \tag{3.2.3}
$$

### 3.3 Calculate the batch normalization partial

$$
\begin{align*}
\frac{\partial \hat x_i} {\partial x_i}
  &= \frac{\partial \frac{x_i - \mu}{\sigma}}{\partial x_i}\\
  &= \frac{\partial [(x_i-\mu)\sigma^{-1}]}{\partial x_i} \\
  &= \frac{\partial x_i-\mu}{\partial x_i}\sigma^{-1}
  -(x_i-\mu)\sigma^{-2}\frac{\partial \sigma}{\partial x_i} \\
  &= \sigma^{-1}(1-\frac{\partial \mu}{\partial x_i})
  -(x_i-\mu)\sigma^{-2}\frac{\partial \sigma}{\partial x_i}
\end{align*}
$$

According to Eq (3.1.1) and (3.2.3), we can got:

$$
\begin{align*}
\frac{\partial \hat x_i} {\partial x_i}
&= \sigma^{-1}(1-\frac{\partial \mu}{\partial x_i})
-(x_i-\mu)\sigma^{-2}\frac{\partial \sigma}{\partial x_i} \\
&= \sigma^{-1}(1-1/m) - (x_i-\mu)\sigma^{-2}(x_i-\mu)/(m\sigma)\\
&=\frac{m-1}{m\sigma}-\frac{(x_i-\mu)^2}{m\sigma^3}
\end{align*}
$$

here is the bottom
