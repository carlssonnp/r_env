-   [Conceptual](#conceptual)
    -   [Question 1](#question-1)
    -   [Question 2](#question-2)
    -   [Question 3](#question-3)
    -   [Question 4](#question-4)
    -   [Question 5](#question-5)
    -   [Question 6](#question-6)
    -   [Question 6](#question-6-1)

    library(ISLR)
    library(tools)
    library(ggplot2)

## Conceptual

### Question 1

#### a

Best subset will have the smallest training RSS, as for a given number
of variables in the model, the models considered by forward / backward
stepwise selection will be a subset of the models chosen by best subset
selection. Best subset therefore looks at every model that forward /
backward stepwise selection looks at, in addition to other models not
examined by the other two procedures. Since it is possible that some of
these other models have lower training RSS, the model chosen by best
subset will have RSS equal to or lower than the models chosen by
forward/backward stepwise selection.

#### b

This is impossible to answer; best subset might have picked a model that
is overfit to the data, since it has more candidate models to choose
from, in which case the other two methods would have better test RSS, or
it might have picked a model that actually describes the
response-predictor relationship better than the other two methods, in
which case it would have better test RSS than the other methods.

#### c

##### i

True

##### ii

True

##### iii

False

##### iv

False

##### v

False

### Question 2

#### a

`iii` is correct because lasso gives biased estimates of the regression
coefficients, but they have lower variance than the OLS estimates.
Depending on the balances of these two competing factors, the prediction
accuracy could be worse or better.

#### b

`iii`, for the same reasons as a.

#### c

`ii` as non-linear methods are more flexible and hence have lower bias
but higher variance.

### Question 3

#### a

`iv`. The training error will always decrease because as we increase
`s`, we decrease the effect of the inequality constraint so that the
minimization of the sum of squared errors dominates the objective
function. As `s` approaches `infinity`, there is no inequality
constraint, so the objective purely minimizes the sum of squared errors,
which is the RSS.

#### b

`ii`. The test error will initially decrease, as the decrease in bias
will outweigh the increase in variance. As we continue to increase `s`,
however, the increase in variance will outweigh the decrease in bias and
the test error will start to increase.

#### c

`iii`. Variance always increases as model flexibility increases, so
increasing `s` will always increase variance.

#### d

`iv`. Bias always decreases as model flexibility increases, so
increasing `s` will always decrease bias.

#### e

`v`. The irreducible error is constant, as this is a property of the
test point rather than the trained model.

### Question 4

#### a

`iii`. The training error will always increase because as we increase
*λ*, we increase the effect of the inequality constraint so that the
minimization of the sum of squared errors plays less of a role in the
objective function. As *λ* approaches `infinity`, all the coefficients
except the intercept approach 0, and assuming the predictors have been
centered to have mean 0 the train RSS will simply be the scaled variance
of the target variable, (*ȳ*−*y*<sub>*i*</sub>)<sup>2</sup>

#### b

`ii`. The test error will initially decrease, as the decrease in
variance will outweigh the increase in bias. As we continue to increase
*λ*, however, the increase in bias will outweigh the decrease in
variance and the test error will start to increase.

#### c

`iv`. Variance always decreases as model flexibility decreases, so
increasing *λ* will always decrease variance.

#### d

`iii`. Bias always increases as model flexibility decreases, so
increasing *λ* will always increase bias.

#### e

`v`. The irreducible error is constant, as this is a property of the
test point rather than the trained model.

### Question 5

#### a

$\displaystyle\min\_{\beta\_1,\beta\_2} \displaystyle\sum\_{i=1}^{2}(\beta\_1x\_{i1} + \beta\_2x\_{i2} - y\_i)^2 + \displaystyle\sum\_{j=1}^{2}\beta\_j^2$

#### b

Differentiate with respect to *β*<sub>1</sub> and set equal to zero to
solve for beta:

$2\displaystyle\sum\_{i=1}^{2}x\_{i1}(\beta\_1x\_{i1} + \beta\_2x\_{i2} - y\_i) + 2\lambda\beta\_1 = 0$

$\displaystyle\sum\_{i=1}^{2}(x\_{i1}^2\beta\_1 + x\_{i1}^2\beta\_2 - y\_i) + \lambda\beta\_1 = 0$

$\displaystyle\sum\_{i=1}^{2}x\_{i1}^2\beta\_1 + \lambda\beta\_1 = \displaystyle\sum\_{i=1}^{2}-x\_{i1}^2\beta\_2 + y\_ix\_i$

$\beta\_1 = \frac{\displaystyle\sum\_{i=1}^{2}-x\_{i1}^2\beta\_2 + y\_ix\_i}{\lambda + \displaystyle\sum\_{i=1}^{2}x\_{i1}^2}$

Similarly we find that

$\beta\_2 = \frac{\displaystyle\sum\_{i=1}^{2}-x\_{i1}^2\beta\_1 + y\_ix\_i}{\lambda + \displaystyle\sum\_{i=1}^{2}x\_{i1}^2}$

The symmetry in these expressions leads us to conclude that
*β*<sub>1</sub> = *β*<sub>2</sub>

#### c

$\displaystyle\min\_{\beta\_1,\beta\_2} \displaystyle\sum\_{i=1}^{2}(\beta\_1x\_{i1} + \beta\_2x\_{i2} - y\_i)^2 + \lambda\displaystyle\sum\_{j=1}^{2}|\beta\_j|$

#### d

This part is easier if we rewrite the above in its alternative
formulation,

$\displaystyle\min\_{\beta\_1,\beta\_2} \displaystyle\sum\_{i=1}^{2}(\beta\_1x\_{i1} + \beta\_2x\_{i2} - y\_i)^2\\\\s.t.\\\displaystyle\sum\_{j=1}^{2}|\beta\_j| \le t$

The level curves of the function we are minimizing are given by

*β*<sub>1</sub>*x*<sub>11</sub> + *β*<sub>2</sub>*x*<sub>11</sub> − *y*<sub>1</sub> + *β*<sub>1</sub>*x*<sub>21</sub> + *β*<sub>2</sub>*x*<sub>21</sub> − *y*<sub>2</sub> = *C*

*β*<sub>1</sub> + *β*<sub>2</sub> = *C*

Which implies that the level curves are lines in the
*β*<sub>1</sub>, *β*<sub>2</sub> space. If *λ* is non-zero, this implies
that the inequality constraint is binding, or that the inequality is in
fact an equality. This means that the solution must lie on one of the
edges of the diamond represented by
$\displaystyle\sum\_{j=1}^{2}|\beta\_j| \le t$. Since these edges are
also straight lines, they touch the level curve at infinitely many
points.

### Question 6

First I will derive a more general form of the ridge regression
estimates when the design matrix is orthogonal. I assume that the
predictors have been centered to have mean 0 as well, so that the
estimate for *β*<sub>0</sub> is equal to *ŷ* for the OLS estimates,
ridge estimates, and lass estimates. The vector *β* that follows is then
the vector of coefficients with the first element removed.

Least squares loss function:

min<sub>*β*</sub>(*y*−*X**β*)<sup>*T*</sup>(*y*−*X**β*)

min<sub>*β*</sub>(*y*<sup>*T*</sup>−*β*<sup>*T*</sup>*X*<sup>*T*</sup>)(*y*−*X**β*)

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − *y*<sup>*T*</sup>*X**β* − *β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β*

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − 2*β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β*

Differentiate with respect to *β* and set equal to zero:

 − 2*X*<sup>*T*</sup>*y* + 2*X*<sup>*T*</sup>*X**β* = 0

*β* = (*X*<sup>*T*</sup>*X*)<sup>−1</sup>*X*<sup>*T*</sup>*y*

Since X is an orthogonal matrix, *X*<sup>*T*</sup>*X* = *I* and
(*X*<sup>*T*</sup>*X*)<sup>−1</sup> = *I*

Then *β* = *X*<sup>*T*</sup>*y*

Ridge regression loss function:

min<sub>*β*</sub>(*y*−*X**β*)<sup>*T*</sup>(*y*−*X**β*) + *λ**β*<sup>*T*</sup>*β*

min<sub>*β*</sub>(*y*<sup>*T*</sup>−*β*<sup>*T*</sup>*X*<sup>*T*</sup>)(*y*−*X**β*) + *λ**β*<sup>*T*</sup>*β*

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − *y*<sup>*T*</sup>*X**β* − *β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β* + *λ**β*<sup>*T*</sup>*β*

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − 2*β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β* + *λ**β*<sup>*T*</sup>*β*

Differentiate with respect to *β* and set equal to zero:

 − 2*X*<sup>*T*</sup>*y* + 2*X*<sup>*T*</sup>*X**β* + 2*λ**β* = 0

*β* = (*X*<sup>*T*</sup>*X*+*λ**I*)<sup>−1</sup>*X*<sup>*T*</sup>*y*

Since X is an orthogonal matrix,
*X*<sup>*T*</sup>*X* + *λ**I* = (1+*λ*)*I* and
$(X^TX + \lambda I)^{-1} = \frac{1}{1 + \lambda}I$

Then
$\beta = \frac{X^Ty}{1 + \lambda} = \frac{\beta\_{OLS}}{1 + \lambda}$

Lasso loss function :

min<sub>*β*</sub>(*y*−*X**β*)<sup>*T*</sup>(*y*−*X**β*) + *λ*||*β*||<sub>1</sub>

min<sub>*β*</sub>(*y*<sup>*T*</sup>−*β*<sup>*T*</sup>*X*<sup>*T*</sup>)(*y*−*X**β*) + *λ*||*β*||<sub>1</sub>

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − *y*<sup>*T*</sup>*X**β* − *β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β* + *λ*||*β*||<sub>1</sub>

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − 2*β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*X*<sup>*T*</sup>*X**β* + *λ*||*β*||<sub>1</sub>

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − 2*β*<sup>*T*</sup>*X*<sup>*T*</sup>*y* + *β*<sup>*T*</sup>*β* + *λ*||*β*||<sub>1</sub>

min<sub>*β*</sub>*y*<sup>*T*</sup>*y* − 2*β*<sup>*T*</sup>*β*<sub>*o**l**s*</sub> + *β*<sup>*T*</sup>*β* + *λ*||*β*||<sub>1</sub>

Take an element `i` of the gradient and set equal to 0. We refer to the
coefficient corresponding to this as *β* now, which is a scalar rather
than a vector.

There are two cases:

1.  *β*<sub>*o**l**s*</sub> ≥ 0, in which case *β* ≥ 0 so as to minimize
    the loss function.

Then  − 2*β*<sub>*o**l**s*</sub> + 2*β* + *λ* = 0

$\beta = \beta\_{ols} - \frac{\lambda}{2}$

Since we assumed that *β* ≥ 0, we take only the positive part of this
equation:

$\beta = (\beta\_{ols} - \frac{\lambda}{2})^+$

$\beta = (sign(\beta\_{ols})|\beta\_{ols}| - \frac{\lambda}{2})^+$

1.  *β*<sub>*o**l**s*</sub> ≤ 0, in which case *β* ≤ 0 so as to minimize
    the loss function.

Then  − 2*β*<sub>*o**l**s*</sub> + 2*β* − *λ* = 0

$\beta = \beta\_{ols} + \frac{\lambda}{2}$

Since we assumed that *β* ≤ 0, we take only the negative part of this
equation:

$\beta = (\beta\_{ols} + \frac{\lambda}{2})^-$

$\beta = sign(|\beta\_{ols})(|\beta\_{ols}| - \frac{\lambda}{2})^+$

We get the same equation for *β* in both cases.

### Question 6

I answer this using the general form of the ridge regression / lasso
estimates in the case of a centered variable with magnitude 1.

#### a

  

    generate_response <- function(df, population_coefficient) {
      df$y <- 10 + population_coefficient * df$x + df$eps
      df
    }

    compare_theoretical_to_simulation <- function(df, betas, method = c("ridge", "lasso")) {

      lambda <- 0.5

      ols_model <- lm(y ~ x, data = df)
      ols_coef <- coef(ols_model)[[2]]

      method <- match.arg(method)
      regularizer <- switch(
        method,
        ridge = function(beta) beta ^ 2,
        lasso = function(beta) abs(beta)
      )

      scaling_factor <- switch(
        method,
        ridge = function(lambda, beta_ols) beta_ols / (1 + lambda),
        lasso = function(lambda, beta_ols) {
          if (abs(beta_ols) >= lambda / 2) {
            sign(beta_ols) * (abs(beta_ols) - lambda / 2)
          } else {
            0
          }
        }
      )

      loss <- sapply(
        betas,
        function(beta, df, lambda, regularizer) {
          sum((df$y - mean(df$y) - beta * df$x) ^ 2) + lambda * regularizer(beta)
        },
        df = df,
        lambda = lambda,
        regularizer = regularizer
      )

      df_loss <- data.frame(beta = betas, loss = loss)
      df_vertical_lines <- data.frame(
        estimates = c(ols_coef, scaling_factor(lambda, ols_coef)),
        method = c("ols", method)
      )

      plot_results(df_loss, df_vertical_lines)

    }

    plot_results <- function(df_loss, df_vertical_lines) {
      ggplot2::ggplot(data = df_loss) +
        ggplot2::geom_point(ggplot2::aes(x = beta, y = loss)) +
        ggplot2::geom_vline(data = df_vertical_lines, ggplot2::aes(xintercept = estimates, color = tools::toTitleCase(method))) +
        ggplot2::labs(x = expression(beta), y = "Loss", color = "Method")
    }

    nrows <- 10000

    set.seed(1)
    df <- data.frame(x = rnorm(nrows), eps = rnorm(nrows, sd = 0.001))
    df$x <- df$x / sqrt(sum(df$x^2))
    df <- generate_response(df, population_coefficient = 1)

    betas <- seq(-1, 3, length = 10000)

    plot_ridge <- compare_theoretical_to_simulation(df, betas, method = "ridge")
    plot_ridge <- plot_ridge +
      ggplot2::ggtitle(expression(Loss~versus~beta~ridge~regression))

    plot_ridge

![](exercises_files/figure-markdown_strict/question_6_a-1.png)

We see that the theoretical value of *β*<sub>*r**i**d**g**e*</sub>
aligns well with the minimum of the loss in the simulation.

#### b

We will consider three cases here:

1.  *β*<sub>*o**l**s*</sub> &lt; 0 and
    $\beta\_{ols} &lt; -\frac{\lambda}{2} 0$
2.  *β*<sub>*o**l**s*</sub> &gt; 0 and
    $\beta\_{ols} &gt; \frac{\lambda}{2} 0$
3.  *β*<sub>*o**l**s*</sub> &gt; 0 and
    $\beta\_{ols} &lt; \frac{\lambda}{2} 0$

<!-- -->

    param_list <- list(
      lasso_1 = list(
        betas = seq(-1, 3, length = 10000),
        population_coefficient = 1,
        title = expression(Loss~versus~beta~lasso~with~beta[ols]>0~and~beta[ols]>lambda/2)
      ),
      lasso_2 = list(
        betas = seq(-3, 1, length = 10000),
        population_coefficient = -1,
        title = expression(Loss~versus~beta~lasso~with~beta[ols]<0~and~paste("|", beta[ols], "|")>lambda/2)
      ),
      lasso_3 = list(
        betas = seq(-0.15, 0.15, length = 10000),
        population_coefficient = 0.01,
        title = expression(Loss~versus~beta~lasso~with~beta[ols]>0~and~beta[ols]<lambda/2)
      )
    )

    for (params in param_list) {
      df_lasso <- generate_response(df, params$population_coefficient)
      plot <- compare_theoretical_to_simulation(df_lasso, params$beta, method = "lasso")
      plot <- plot + ggplot2::labs(title = params$title)
      print(plot)
    }

![](exercises_files/figure-markdown_strict/question_6_b-1.png)![](exercises_files/figure-markdown_strict/question_6_b-2.png)![](exercises_files/figure-markdown_strict/question_6_b-3.png)
\## Applied

\`\`
