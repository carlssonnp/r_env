library(ggplot2)
library(ISLR)
library(MASS)

df_default <- Default


output_test_set_results <- function(df, student = FALSE) {
  train_idx <- sample(nrow(df), nrow(df) %/% 2)

  train <- df[train_idx, ]
  test <- df[-train_idx, ]

  if(isTRUE(student)) {
    mod <- glm(default ~ income + balance + student, data = train, family = "binomial")
  } else {
    mod <- glm(default ~ income + balance, data = train, family = "binomial")
  }

  preds <- predict(mod, test, type = "response")

  labels <- ifelse(preds >= 0.5, "Yes", "No")

  mean(labels == test$default)
}

lapply(1:5, function(idx) output_test_set_results(df_default))

# [[1]]
# [1] 0.973
#
# [[2]]
# [1] 0.9734
#
# [[3]]
# [1] 0.9738
#
# [[4]]
# [1] 0.9722
#
# [[5]]
# [1] 0.9762


# Pretty close to each other, although they differ somewhat

lapply(1:5, function(idx) output_test_set_results(df_default, student = TRUE))

# [[1]]
# [1] 0.9728
#
# [[2]]
# [1] 0.9752
#
# [[3]]
# [1] 0.974
#
# [[4]]
# [1] 0.975
#
# [[5]]
# [1] 0.9742


# No evidence it is better with student included


summary(glm(default ~ income + balance, family = "binomial", data = df_default))

# Call:
# glm(formula = default ~ income + balance, family = "binomial",
#     data = df_default)
#
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)
# (Intercept) -1.154e+01  4.348e-01 -26.545  < 2e-16 ***
# income       2.081e-05  4.985e-06   4.174 2.99e-05 ***
# balance      5.647e-03  2.274e-04  24.836  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# (Dispersion parameter for binomial family taken to be 1)
#
#     Null deviance: 2920.6  on 9999  degrees of freedom
# Residual deviance: 1579.0  on 9997  degrees of freedom
# AIC: 1585
#
# Number of Fisher Scoring iterations: 8


logistic_regression_subset <- function(df, index) {
  df <- df[index, ]

  logistic_regression_model <- glm(default ~ income + balance, data = df, family = "binomial")

  coef(logistic_regression_model)
}


# manually

estimates <- vector("list", length = 100)
for (i in seq_along(estimates)) {
  samp_idx <- sample(nrow(df_default), replace = TRUE)
  estimates[[i]] <- logistic_regression_subset(df_default, samp_idx)
}

sd(sapply(estimates, function(estimate) estimate[[1]]))
# [1] 0.4444479
sd(sapply(estimates, function(estimate) estimate[[2]]))
# [1] 4.915305e-06
sd(sapply(estimates, function(estimate) estimate[[3]]))
# [1] 0.0002450696

# Very close to estimates

# use boot::boot

boot::boot(df_default, logistic_regression_subset, R=100)

# original        bias     std. error
# t1* -1.154047e+01  9.684811e-02 4.463521e-01
# t2*  2.080898e-05 -6.314839e-07 4.917093e-06
# t3*  5.647103e-03 -5.268144e-05 2.307704e-04

full_logistic_regression_model <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = "binomial")

cost <- function(y_true, y_pred) {
  y_true == ifelse(y_pred >= 0.5, 1, 0)
}

loocv_estimate <- boot::cv.glm(Weekly, full_logistic_regression_model, cost=cost)$delta[[1]]
loocv_estimate
# [1] 0.5500459


# do it manually

estimates <- rep(0, nrow(Weekly))

for (i in seq_along(estimates)) {
  train <- Weekly[-i, ]
  test <- Weekly[i, ]

  model <- glm(Direction ~ Lag1 + Lag2, data = train, family = "binomial")

  pred <- predict(model, test, type = "response")

  estimates[[i]] <- test$Direction == ifelse(pred >= 0.5, "Up", "Down")

}

mean(estimates)
# [1] 0.5500459

# Exact same


x <- rnorm(100)
y <- x - 2 * x ^ 2 + rnorm(100)

df <- data.frame(x = x, y = y)

ggplot2::ggplot(data = df) +
  ggplot2::geom_point(ggplot2::aes(x = x, y = y))

# Highly non-linear

polys <- 1:4
loocvs <- rep(0, length(polys))

for (i in polys) {
  model <- glm(y ~ poly(x, i), data = df)
  loocvs[[i]] <- cv.glm(df, model)$delta[[1]]
}

loocvs
# [1] 5.207325 1.087738 1.098772 1.119041


# No randomness here since it is loocv

summary(model)

# Call:
# glm(formula = y ~ poly(x, i), data = df)
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)  -1.3978     0.1044 -13.392   <2e-16 ***
# poly(x, i)1   2.9463     1.0438   2.823   0.0058 **
# poly(x, i)2 -19.4114     1.0438 -18.597   <2e-16 ***
# poly(x, i)3  -0.4924     1.0438  -0.472   0.6382
# poly(x, i)4   0.1597     1.0438   0.153   0.8788
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# (Dispersion parameter for gaussian family taken to be 1.089497)
#
#     Null deviance: 489.25  on 99  degrees of freedom
# Residual deviance: 103.50  on 95  degrees of freedom
# AIC: 299.23
#
# Number of Fisher Scoring iterations: 2


# Agrees with loocv scores; starts to overfit as more variables are added


df_boston <- Boston

mean(df_boston$medv)
# [1] 22.53281

sd(df_boston$medv) / sqrt(nrow(df_boston))
# [1] 0.4088611


r <- 1000
estimates <- rep(0, r)

for (i in seq_along(estimates)) {
  samp_idx <- sample(nrow(df_boston), replace = TRUE)
  estimates[[i]] <- mean(df_boston[samp_idx, "medv"])
}

sd(estimates)
# [1] 0.4184424

# Close

# do it using boot

my_mean <- function(df, index) {
  mean(df[index, "medv"])
}

boot::boot(df_boston, my_mean, R = r)

# ORDINARY NONPARAMETRIC BOOTSTRAP
#
#
# Call:
# boot::boot(data = df_boston, statistic = my_mean, R = r)
#
#
# Bootstrap Statistics :
#     original      bias    std. error
# t1* 22.53281 0.005683597   0.4047507


# 22.53281 +/- 2* 0.4184424


t.test(df_boston$medv)

# One Sample t-test

# data:  df_boston$medv
# t = 55.111, df = 505, p-value < 2.2e-16
# alternative hypothesis: true mean is not equal to 0
# 95 percent confidence interval:
# 21.72953 23.33608
# sample estimates:
# mean of x
# 22.53281


# Agrees well t.test interval

median(df_boston$medv)
# [1] 21.2


r <- 10000
estimates <- rep(0, r)

for (i in seq_along(estimates)) {
  sample_idx <- sample(nrow(df_boston), replace = TRUE)
  estimates[[i]] <- median(df_boston[sample_idx, "medv"])
}

sd(estimates)
# [1] 0.378547


quantile(df_boston$medv, 0.1)
# 10%
# 12.75


r <- 10000
estimates <- rep(0, r)

for (i in seq_along(estimates)) {
  sample_idx <- sample(nrow(df_boston), replace = TRUE)
  estimates[[i]] <- quantile(df_boston[sample_idx, "medv"], 0.1)
}

sd(estimates)

# [1] 0.5008761



# central limit theorem

means <- rep(0, 10000)


for (i in seq_along(means)) {
  x1 <- rnorm(100, 5)
  x2 <- rpois(100, 2)
  x3 <- 4 * rpois(100, 2)
  x4 <- 1:100 * rpois(100, 2)

  x <- c(x1, x2, x3, x4)

  means[[i]] <- mean(x)
}


mean(means)
# [1] 3.500301

sd(means)
# [1] 0.1944807

hist(means)


theoretical_mean <- (100 * 5 + 100 * 2 + 100 * 8) / 300
# [1] 5

theoretical_sd <- sqrt((100 * 1 + 100 * 2 + 100 * 32) / 300^2)

# [1] 0.1972027


# Distribution of means is normal, even though they come from different distributions!
