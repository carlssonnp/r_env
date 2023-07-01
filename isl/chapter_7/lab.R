library(ISLR)
library(ggplot2)


# Investigating orthogonal inputs
lm_poly <- lm(Sepal.Width ~ poly(Sepal.Length, 3), data = iris)

lm_1 <- lm(Sepal.Width ~ poly(Sepal.Length, 3)[, 1], data = iris)
lm_2 <- lm(Sepal.Width ~ poly(Sepal.Length, 3)[, 2], data = iris)
lm_3 <- lm(Sepal.Width ~ poly(Sepal.Length, 3)[, 3], data = iris)


summary(lm_poly)

# Call:
# lm(formula = Sepal.Width ~ poly(Sepal.Length, 3), data = iris)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -1.17219 -0.23769 -0.00581  0.27359  1.34285
#
# Coefficients:
#                        Estimate Std. Error t value Pr(>|t|)
# (Intercept)             3.05733    0.03479  87.869   <2e-16 ***
# poly(Sepal.Length, 3)1 -0.62552    0.42614  -1.468   0.1443
# poly(Sepal.Length, 3)2  0.82430    0.42614   1.934   0.0550 .
# poly(Sepal.Length, 3)3  0.85028    0.42614   1.995   0.0479 *
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4261 on 146 degrees of freedom
# Multiple R-squared:  0.06337,   Adjusted R-squared:  0.04412
# F-statistic: 3.292 on 3 and 146 DF,  p-value: 0.02239


summary(lm_1)

# Call:
# lm(formula = Sepal.Width ~ poly(Sepal.Length, 3)[, 1], data = iris)

# Residuals:
#     Min      1Q  Median      3Q     Max
# -1.1095 -0.2454 -0.0167  0.2763  1.3338
#
# Coefficients:
#                            Estimate Std. Error t value Pr(>|t|)
# (Intercept)                 3.05733    0.03546   86.22   <2e-16 ***
# poly(Sepal.Length, 3)[, 1] -0.62552    0.43430   -1.44    0.152
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4343 on 148 degrees of freedom
# Multiple R-squared:  0.01382,   Adjusted R-squared:  0.007159
# F-statistic: 2.074 on 1 and 148 DF,  p-value: 0.1519


summary(lm_2)
#
# Call:
# lm(formula = Sepal.Width ~ poly(Sepal.Length, 3)[, 2], data = iris)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -1.07851 -0.25227 -0.02143  0.27781  1.39612
#
# Coefficients:
#                            Estimate Std. Error t value Pr(>|t|)
# (Intercept)                 3.05733    0.03528  86.666   <2e-16 ***
# poly(Sepal.Length, 3)[, 2]  0.82430    0.43206   1.908   0.0583 .
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4321 on 148 degrees of freedom
# Multiple R-squared:  0.024,     Adjusted R-squared:  0.01741
# F-statistic:  3.64 on 1 and 148 DF,  p-value: 0.05835


summary(lm_3)

# Call:
# lm(formula = Sepal.Width ~ poly(Sepal.Length, 3)[, 3], data = iris)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -1.09882 -0.28410  0.00641  0.28258  1.29826
#
# Coefficients:
#                            Estimate Std. Error t value Pr(>|t|)
# (Intercept)                 3.05733    0.03525   86.73   <2e-16 ***
# poly(Sepal.Length, 3)[, 3]  0.85028    0.43172    1.97   0.0508 .
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4317 on 148 degrees of freedom
# Multiple R-squared:  0.02554,   Adjusted R-squared:  0.01896
# F-statistic: 3.879 on 1 and 148 DF,  p-value: 0.05076


df_wage <- Wage

fit <- lm(wage ~ poly(age, 4), data = df_wage)

summary(fit)

# Call:
# lm(formula = wage ~ poly(age, 4), data = df_wage)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -98.707 -24.626  -4.993  15.217 203.693
#
# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)
# (Intercept)    111.7036     0.7287 153.283  < 2e-16 ***
# poly(age, 4)1  447.0679    39.9148  11.201  < 2e-16 ***
# poly(age, 4)2 -478.3158    39.9148 -11.983  < 2e-16 ***
# poly(age, 4)3  125.5217    39.9148   3.145  0.00168 **
# poly(age, 4)4  -77.9112    39.9148  -1.952  0.05104 .

age_grid <- do.call(seq, as.list(range(df_wage$age)))

preds <- predict(fit, data.frame(age = age_grid), se = TRUE)

df_preds <- data.frame(
  predictions = preds$fit,
  upper = preds$fit + 2 * preds$se.fit,
  lower = preds$fit - 2 * preds$se.fit,
  age = age_grid
)

df_preds <- tidyr::pivot_longer(df_preds, cols = c("predictions", "upper", "lower"))


plt <- ggplot2::ggplot(data = df_preds) +
  ggplot2::geom_smooth(ggplot2::aes(x = age, y = value, color = name))


mods <- lapply(
  1:5,
  function(i) {
    lm(wage ~ poly(age, i), data = df_wage)
  }
)

do.call(anova, mods)

# Analysis of Variance Table
#
# Model 1: wage ~ poly(age, i)
# Model 2: wage ~ poly(age, i)
# Model 3: wage ~ poly(age, i)
# Model 4: wage ~ poly(age, i)
# Model 5: wage ~ poly(age, i)
#   Res.Df     RSS Df Sum of Sq        F    Pr(>F)
# 1   2998 5022216
# 2   2997 4793430  1    228786 143.5931 < 2.2e-16 ***
# 3   2996 4777674  1     15756   9.8888  0.001679 **
# 4   2995 4771604  1      6070   3.8098  0.051046 .
# 5   2994 4770322  1      1283   0.8050  0.369682
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


logistic_fit <- glm(I(wage > 250) ~ poly(age, 4), data = df_wage, family = "binomial")
preds <- predict(logistic_fit, data.frame(age = age_grid), se = TRUE)


df_preds <- data.frame(
  predictions = preds$fit,
  upper = preds$fit + 2 * preds$se.fit,
  lower = preds$fit - 2 * preds$se.fit,
  age = age_grid
)

df_preds <- tidyr::pivot_longer(df_preds, cols = c("predictions", "upper", "lower"))

df_preds$value <- 1 / (1 + exp(-df_preds$value))

plt <- ggplot2::ggplot(data = df_preds) +
  ggplot2::geom_smooth(ggplot2::aes(x = age, y = value, color = name))




# Splines

cubic_spline_model <- lm(wage ~ splines::bs(age, knots = c(25, 40, 60)), data = df_wage)

preds_df <- data.frame(
  preds = predict(cubic_spline_model, data.frame(age = age_grid)),
  age = age_grid
)

ggplot(data = preds_df) +
  geom_smooth(aes(x = age, y = preds))

natural_spline_model <- lm(wage ~ splines::ns(age, knots = c(25, 40, 60)), data = df_wage)

preds_df <- data.frame(
  preds = predict(natural_spline_model, data.frame(age = c(5, age_grid, 100))),
  age = c(5, age_grid, 100)
)

ggplot(data = preds_df) +
  geom_smooth(aes(x = age, y = preds))


smoothed_spline <- smooth.spline(x = df_wage$age, y = df_wage$wage, df = 16)
smoothed_spline <- smooth.spline(x = df_wage$age, y = df_wage$wage, cv = TRUE)


local_regression <- loess(wage ~ age, data = df_wage, span = 0.75)


resids <- residuals(fit)
studentized_resids <- resids / sqrt((sum((resids - mean(resids))^2)) / 148)

empirical_results <- rank(studentized_resids) / length(studentized_resids)
theoretical_results <- qnorm(empirical_results)


df <- data.frame(theoretical = theoretical_results, empirical = studentized_resids)

ggplot(df) +
  geom_point(aes(y = empirical, x = theoretical)) +
  geom_abline(intercept = 0, slope = 1)



gam1 <- lm(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)


gam_3 = gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage)

gam_1 <- gam(wage ~ s(age, 5) + education, data = Wage)

gam_2 <- gam(wage ~ year + s(age, 5) + education, data = Wage)


anova(gam_1, gam_2, gam_3, test="F")

# Analysis of Deviance Table
#
# Model 1: wage ~ s(age, 5) + education
# Model 2: wage ~ year + s(age, 5) + education
# Model 3: wage ~ s(year, 4) + s(age, 5) + education
#   Resid. Df Resid. Dev Df Deviance       F    Pr(>F)
# 1      2990    3711731
# 2      2989    3693842  1  17889.2 14.4771 0.0001447 ***
# 3      2986    3689770  3   4071.1  1.0982 0.3485661
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# No evidence of non-linear relationship with year

gam_lo <- gam(wage ~ lo(year, age, span = 0.5), data = Wage)


cross_validate <- function(df, k) {
  indices <- rep(seq(k), length.out = nrow(df))
  indices <- sample(indices)

  split(df, indices)
}

df_list <- cross_validate(Wage, 5)

res <- matrix(0, 5, 10)
for (i in seq_along(df_list)) {
  train <- do.call(rbind, df_list[-i])
  test <- df_list[[i]]
  for (j in seq(10)) {
    mod <- lm(wage ~ poly(age, j), data = train)
    preds <- predict(mod, test)
    res[i, j] <- mean((preds - test$wage) ^ 2)
  }
}


res <- apply(res, 2, mean)

which.min(res)
# [1] 4


# 4th degree polynomial seems best

# compare to anova
print(summary(lm(wage ~ poly(age, 10), data = Wage)))

# Call:
# lm(formula = wage ~ poly(age, 10), data = Wage)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -100.38  -24.45   -4.97   15.49  199.61
#
# Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)
# (Intercept)      111.7036     0.7283 153.369  < 2e-16 ***
# poly(age, 10)1   447.0679    39.8924  11.207  < 2e-16 ***
# poly(age, 10)2  -478.3158    39.8924 -11.990  < 2e-16 ***
# poly(age, 10)3   125.5217    39.8924   3.147  0.00167 **
# poly(age, 10)4   -77.9112    39.8924  -1.953  0.05091 .
# poly(age, 10)5   -35.8129    39.8924  -0.898  0.36940
# poly(age, 10)6    62.7077    39.8924   1.572  0.11607
# poly(age, 10)7    50.5498    39.8924   1.267  0.20520
# poly(age, 10)8   -11.2547    39.8924  -0.282  0.77787
# poly(age, 10)9   -83.6918    39.8924  -2.098  0.03599 *
# poly(age, 10)10    1.6240    39.8924   0.041  0.96753
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 39.89 on 2989 degrees of freedom
# Multiple R-squared:  0.08912,   Adjusted R-squared:  0.08607
# F-statistic: 29.24 on 10 and 2989 DF,  p-value: < 2.2e-16


final_model <- lm(wage ~ poly(age, 4), data = Wage)

age_grid <- do.call(seq, as.list(range(Wage$age)))


df <- data.frame(age = age_grid, preds = predict(final_model, list(age = age_grid)))

ggplot(df) +
  geom_point(aes(x = age, y = preds))

df_list <- cross_validate(Wage, 5)

# I did equal intervals in terms of numbers of observations, rather than equal in terms
# of bin width
res <- matrix(0, 5, 9)
for (i in seq_along(df_list)) {
  train <- do.call(rbind, df_list[-i])
  test <- df_list[[i]]
  for (j in seq(2, 10)) {
    breaks <- seq(0, 1, by = 1 / j)
    breaks <- quantile(train$age, breaks)
    train$age_interval <- cut(train$age, breaks, include.lowest = TRUE)
    test$age_interval <- cut(test$age, breaks, include.lowest = TRUE)
    mod <- lm(wage ~ age_interval, data = train)
    preds <- predict(mod, test)
    res[i, j - 1] <- mean((preds - test$wage) ^ 2)
  }
}


res <- apply(res, 2, mean)

which.min(res)
# 8


df_wage <- Wage
#
# breaks <- seq(0, 1, by = 1 / 8)
# breaks <- quantile(df_wage$age, breaks)
df_wage$age_interval <- cut(df_wage$age, 8)

final_model <- lm(wage ~ age_interval, data = df_wage)

df <- data.frame(age = df_wage$age, preds = fitted(final_model))

ggplot(df) +
  geom_line(aes(x = age, y = preds))



model <- lm(nox ~ poly(dis, 3), data = Boston)

Call:
lm(formula = nox ~ poly(dis, 3), data = Boston)

# Residuals:
#       Min        1Q    Median        3Q       Max
# -0.121130 -0.040619 -0.009738  0.023385  0.194904
#
# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)
# (Intercept)    0.554695   0.002759 201.021  < 2e-16 ***
# poly(dis, 3)1 -2.003096   0.062071 -32.271  < 2e-16 ***
# poly(dis, 3)2  0.856330   0.062071  13.796  < 2e-16 ***
# poly(dis, 3)3 -0.318049   0.062071  -5.124 4.27e-07 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.06207 on 502 degrees of freedom
# Multiple R-squared:  0.7148,    Adjusted R-squared:  0.7131
# F-statistic: 419.3 on 3 and 502 DF,  p-value: < 2.2e-16


# Looks pretty good

full_model <- lm(nox ~ poly(dis, 10), data = Boston)


summary(full_model)

# Call:
# lm(formula = nox ~ poly(dis, 10), data = Boston)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -0.12978 -0.03816 -0.01015  0.02420  0.19694
#
# Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)
# (Intercept)      0.554695   0.002705 205.092  < 2e-16 ***
# poly(dis, 10)1  -2.003096   0.060839 -32.925  < 2e-16 ***
# poly(dis, 10)2   0.856330   0.060839  14.075  < 2e-16 ***
# poly(dis, 10)3  -0.318049   0.060839  -5.228 2.54e-07 ***
# poly(dis, 10)4   0.033547   0.060839   0.551  0.58161
# poly(dis, 10)5   0.133009   0.060839   2.186  0.02926 *
# poly(dis, 10)6  -0.192439   0.060839  -3.163  0.00166 **
# poly(dis, 10)7   0.169628   0.060839   2.788  0.00550 **
# poly(dis, 10)8  -0.117703   0.060839  -1.935  0.05360 .
# poly(dis, 10)9   0.047947   0.060839   0.788  0.43102
# poly(dis, 10)10 -0.034054   0.060839  -0.560  0.57591
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.06084 on 495 degrees of freedom
# Multiple R-squared:  0.7298,    Adjusted R-squared:  0.7243
# F-statistic: 133.7 on 10 and 495 DF,  p-value: < 2.2e-16


# Looks like up to 3 is a pretty good fit


df_list <- cross_validate(Boston, 5)

res <- matrix(0, 5, 10)
for (i in seq_along(df_list)) {
  train <- do.call(rbind, df_list[-i])
  test <- df_list[[i]]
  for (j in seq(10)) {
    mod <- lm(nox ~ poly(dis, j), data = train)
    preds <- predict(mod, test)
    res[i, j] <- mean((preds - test$nox) ^ 2)
  }
}


res <- apply(res, 2, mean)

which.min(res)

# 3


model <- lm(nox ~ bs(dis, 4), data = Boston)


summary(model)

# Call:
# lm(formula = nox ~ bs(dis, 4), data = Boston)
#
# Residuals:
#       Min        1Q    Median        3Q       Max
# -0.124622 -0.039259 -0.008514  0.020850  0.193891
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)  0.73447    0.01460  50.306  < 2e-16 ***
# bs(dis, 4)1 -0.05810    0.02186  -2.658  0.00812 **
# bs(dis, 4)2 -0.46356    0.02366 -19.596  < 2e-16 ***
# bs(dis, 4)3 -0.19979    0.04311  -4.634 4.58e-06 ***
# bs(dis, 4)4 -0.38881    0.04551  -8.544  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.06195 on 501 degrees of freedom
# Multiple R-squared:  0.7164,    Adjusted R-squared:  0.7142
# F-statistic: 316.5 on 4 and 501 DF,  p-value: < 2.2e-16



df_list <- cross_validate(Boston, 5)

res <- matrix(0, 5, 12)
for (i in seq_along(df_list)) {
  train <- do.call(rbind, df_list[-i])
  test <- df_list[[i]]
  for (j in seq(4, 15)) {
    mod <- lm(nox ~ bs(dis, j), data = train)
    preds <- predict(mod, test)
    res[i, j - 3] <- mean((preds - test$nox) ^ 2)
  }
}


res <- apply(res, 2, mean)

which.min(res)


# R> res
#  [1] 0.003897586 0.003721664 0.003734266 0.003739773 0.003716076 0.003729518
#  [7] 0.003717496 0.003762936 0.003744006 0.003738076 0.003751577 0.003761609

# pretty stable with different numbers of knots




# question 10

train_idx <- sample(nrow(College), nrow(College) %/% 2)

train <- College[train_idx, ]
test <- College[-train_idx, ]


models <- regsubsets(Outstate ~ ., data = train, nvmax = Inf, method = "forward")


plot(models)


full_model <- lm(Outstate ~ Private + Room.Board + Personal + Terminal + perc.alumni + Expend + Grad.Rate, data = train)
gam_model <- lm(Outstate ~ ns(Room.Board, 4) + ns(Personal, 4) + ns(Terminal, 4) + ns(perc.alumni, 4) + ns(Expend, 4) + ns(Grad.Rate, 4), data = train)


preds_full <- predict(full_model, test)

sqrt(mean((preds_full - test$Outstate) ^ 2))
# [1] 2007.731

preds_gam <- predict(gam_model, test)
sqrt(mean((preds_gam - test$Outstate) ^ 2))


# The adjusted r squared is higher with the gam_model, but the test error is higher. Maybe we have overfit

# Can also try with smoothed splines


gam_model <- gam(Outstate ~ s(Room.Board, 4) + s(Personal, 4) + s(Terminal, 4) + s(perc.alumni, 4) + s(Expend, 4) + s(Grad.Rate, 4), data = train)


# Expend and graduation rate are both highly non-linear


x1 <- rnorm(100)
x2 <- rnorm(100)

y <- 10 + 2 * x1 + 3 * x2 + rnorm(100, sd = 0.25)


b1 <- 2

response <- y
for (i in seq(1000)) {
  response <- y - b1 * x1
  mod <- lm(response ~ x2)
  b2 <- mod$coef[[2]]
  response <- y - b2 * x2
  mod <- lm(response ~ x1)
  b1 <- mod$coef[[2]]
}


print(b1)
print(b2)
print(mod$coef[[1]])



x <- matrix(rnorm(10000 * 100), 10000, 100)

coefs <- sample(100)

y <- x %*% coefs + rnorm(10000)


response <- y
betas <- rep(0, 100)
for (i in seq(100)) {
  for (j in seq(100)) {
    response <- y - x[, -j] %*% betas[-j]
    mod <- lm(response ~ x[, j])
    betas[[j]] <- mod$coef[[2]]
  }
}
