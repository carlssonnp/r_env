library(boot)
library(ISLR)
library(ggplot2)

# First, conceptual exercises

n <- 1:100000
p <- 1 - (1 - 1 / n) ^ n

df <- data.frame(n = n, p = p)

ggplot2::ggplot(data = df) +
  ggplot2::geom_smooth(ggplot2::aes(x = n, y = p))


# Approaches a limit as n <- inf

mean_present <- mean(sapply(1:100000, function(idx) (sum(sample(1:100, 100, replace=TRUE) == 4)) > 0))

mean_present

# [1] 0.63448

# Agrees with theoretical results


set.seed(1)

df_auto <- Auto %>%
  dplyr::select(., -name)
len <- nrow(df_auto)

train_idx <- sample(len, len %/% 2)

train_df <- df_auto[train_idx, ]
test_df <- df_auto[-train_idx, ]

lm_order_1 <- lm(mpg ~ horsepower, data = train_df)

mean((test_df$mpg - predict(lm_order_1, test_df)) ^ 2)

# [1] 26.83974

lm_order_2 <- lm(mpg ~ poly(horsepower, 2), data = train_df)

mean((test_df$mpg - predict(lm_order_2, test_df)) ^ 2)
# [1] 19.56785

lm_order_3 <- lm(mpg ~ poly(horsepower, 3), data = train_df)

mean((test_df$mpg - predict(lm_order_3, test_df)) ^ 2)

# [1] 19.62272

# Leave one out cross-validation

lm_order_1 <- glm(mpg ~ horsepower, data = df_auto)

# second arg must be fitted to first arg
loocv <- boot::cv.glm(df_auto, lm_order_1)

loocv$delta
# [1] 24.23151 24.23114

# Corresponds to the LOOCV error


poly_values <- 1:10
res <- rep(0, length(poly_values))

for (i in poly_values) {
  mod <- glm(mpg ~ poly(horsepower, i), data = df_auto)
  res[i] <- cv.glm(df_auto, mod)$delta[[1]]
}


res

# [1] 24.23151 19.24821 19.33498 19.42443 19.03321 18.97864 18.83305 18.96115
# [9] 19.06863 19.49093

poly_values <- 1:10
res <- rep(0, length(poly_values))

for (i in poly_values) {
  mod <- glm(mpg ~ poly(horsepower, i), data = df_auto)
  res[i] <- cv.glm(df_auto, mod, K = 10 )$delta[[1]]
}


res

# [1] 24.44051 19.35140 19.24440 19.29638 18.96664 19.07148 18.86015 18.95843
# [9] 18.83820 19.58445



# The bootstrap


# Apply it to mean

set.seed(1000)
x <- rnorm(100, 2, 1)

sd_estimate <- sd(x)

sd_mean_estimate <- sd_estimate / sqrt(length(x))

sd_mean_estimate
# [1] 0.1095641

# sampled sd_mean
x_list <- replicate(10000, rnorm(100, 2, 1), simplify=FALSE)

sd(sapply(x_list, mean))
# [1] 0.1004659

# True sd mean

1 / sqrt(100)

# [1] 0.1

# bootstrap mean
# manual way

x_list_boot <- replicate(10000, sample(x, replace = TRUE), simplify = FALSE)
sd(sapply(x_list_boot, mean))

# [1] 0.08947643

my_mean <- function(x, index) {
  mean(x[index])
}

boot::boot(x, my_mean, R=10000)

original       bias    std. error
# t1* 2.108887 -0.001443417  0.08920386


# Got much closer results with a different seed



# quick test of variance of ols estimates

x <- rnorm(100)
y <- x + rnorm(100, 0, 0.25)

summary(lm(y ~ x))

# Call:
# lm(formula = y ~ x)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -0.68227 -0.17552  0.01149  0.16920  0.60068
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) -0.01989    0.02498  -0.796    0.428
# x            1.01007    0.02547  39.658   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.2467 on 98 degrees of freedom
# Multiple R-squared:  0.9413,    Adjusted R-squared:  0.9407
# F-statistic:  1573 on 1 and 98 DF,  p-value: < 2.2e-16


coefs <- rep(0, 1000)

for (i in 1:1000) {
  y_new <- x + rnorm(100, 0, 0.25)
  coefs[i] <- coef(lm(y_new ~ x))[["x"]]
}

sd(coefs)
# [1] 0.02610077

# Very close to estimated standard error from one model


alpha_estimate <- function(df, index) {
  x <- df[index, "X"]
  y <- df[index, "Y"]

  (var(y) - cov(x, y)) / (var(x) + var(y) - 2 * cov(x, y))
}


# manual bootstrap

df_portfolio <- Portfolio

n_estimates <- 1000
estimates <- rep(0, n_estimates)

for (i in seq_along(estimates)) {
  samp_idx <- sample(nrow(df_portfolio), replace = TRUE)
  estimates[[i]] <- alpha_estimate(df_portfolio, samp_idx)
}

sd(estimates)
# [1] 0.08876696


# should be same as
boot::boot(df_portfolio, alpha_estimate, R=1000)
# original      bias    std. error
# t1* 0.5758321 0.001353628   0.0928413


df_auto <- Auto

coefficient_estimate <- function(df, index) {
  coef(lm(mpg ~ horsepower, data = df[index, ]))
}

n_estimates <- 1000
estimates <- vector("list", 1000)

for (i in seq_along(estimates)) {
  samp_idx <- sample(nrow(df_auto), replace = TRUE)
  estimates[[i]] <- coefficient_estimate(df_auto, samp_idx)
}


sd(sapply(estimates, function(estimate) estimate[[1]]))
# [1] 0.8806603
sd(sapply(estimates, function(estimate) estimate[[2]]))
# [1] 0.007593812

lm_model <- lm(mpg ~ horsepower, data = df_auto)
print(summary(lm_model))
Call:
lm(formula = mpg ~ horsepower, data = df_auto)

# Residuals:
#      Min       1Q   Median       3Q      Max
# -13.5710  -3.2592  -0.3435   2.7630  16.9240
#
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)
# (Intercept) 39.935861   0.717499   55.66   <2e-16 ***
# horsepower  -0.157845   0.006446  -24.49   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.906 on 390 degrees of freedom
# Multiple R-squared:  0.6059,    Adjusted R-squared:  0.6049
# F-statistic: 599.7 on 1 and 390 DF,  p-value: < 2.2e-16


# The standard errors are higher using the bootstap method, becausee
# 1) The assumptions of the linear model are not met; and
# 2) The linear model assumes you would sample from the same x and just get
 # and new random error term each time. 
