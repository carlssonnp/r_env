# modern applied statistics with S
library(MASS)
library(ISLR)
# companion to applied regression
library(car)

df_boston <- Boston

lm_model <- lm(medv ~ lstat, data=df_boston)

print(summary(lm_model))

# Call:
# lm(formula = medv ~ lstat, data = df_boston)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -15.168  -3.990  -1.318   2.034  24.500
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 34.55384    0.56263   61.41   <2e-16 ***
# lstat       -0.95005    0.03873  -24.53   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 6.216 on 504 degrees of freedom
# Multiple R-squared:  0.5441,    Adjusted R-squared:  0.5432
# F-statistic: 601.6 on 1 and 504 DF,  p-value: < 2.2e-16


# Note that the F-statistic is the square of the t-statistic, on one
# degree of freedom and n-2=504 degrees of freedom.

confint(lm_model)

# 2.5 %     97.5 %
# (Intercept) 33.448457 35.6592247
# lstat       -1.026148 -0.8739505

test_data <- data.frame(
  lstat = seq(5, 15, 5)
)

predict(lm_model, test_data, interval="confidence")
#
# fit      lwr      upr
# 1 29.80359 29.00741 30.59978
# 2 25.05335 24.47413 25.63256
# 3 20.30310 19.73159 20.87461

predict(lm_model, test_data, interval="prediction")
# fit       lwr      upr
# 1 29.80359 17.565675 42.04151
# 2 25.05335 12.827626 37.27907
# 3 20.30310  8.077742 32.52846


# Plot medv and lstat, along with least squares regression line
preds_df <- data.frame(
  predicted_medv = predict(lm_model),
  medv = df_boston$medv, lstat = df_boston$lstat,
  residuals = residuals(lm_model),
  studentized_residuals = rstudent(lm_model),
  leverage_statistic = hatvalues(lm_model)
)

simple_linear_regression_y_x_plot <- ggplot2::ggplot(preds_df) +
  ggplot2::geom_point(ggplot2::aes(x = lstat, y = medv)) +
  ggplot2::geom_line(ggplot2::aes(x = lstat, y = predicted_medv))

ggplot2::ggsave("isl/chapter_3/images/simple_linear_regression_y_x_plot.png", simple_linear_regression_y_x_plot)

residual_plot <- ggplot2::ggplot(preds_df) +
  ggplot2::geom_point(ggplot2::aes(x = predicted_medv, y = studentized_residuals))

ggplot2::ggsave("isl/chapter_3/images/simple_linear_regression_residual_plot.png", residual_plot)

leverage_plot <- ggplot2::ggplot(preds_df) +
  ggplot2::geom_point(ggplot2::aes(x = lstat, y = leverage_statistic))

# quadratic in x, as the formula suggests
ggplot2::ggsave("isl/chapter_3/images/simple_linear_regression_leverage_plot.png", leverage_plot)


# Multiple regresion
lm_model_multiple <- lm(medv ~ lstat + age, data = df_boston)

summary(lm_model_multiple)

# Call:
# lm(formula = medv ~ lstat + age, data = df_boston)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -15.981  -3.978  -1.283   1.968  23.158
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 33.22276    0.73085  45.458  < 2e-16 ***
# lstat       -1.03207    0.04819 -21.416  < 2e-16 ***
# age          0.03454    0.01223   2.826  0.00491 **
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 6.173 on 503 degrees of freedom
# Multiple R-squared:  0.5513,    Adjusted R-squared:  0.5495
# F-statistic:   309 on 2 and 503 DF,  p-value: < 2.2e-16


# compare models using F test

anova(lm_model, lm_model_multiple)

# Analysis of Variance Table
#
# Model 1: medv ~ lstat
# Model 2: medv ~ lstat + age
#   Res.Df   RSS Df Sum of Sq     F   Pr(>F)
# 1    504 19472
# 2    503 19168  1    304.25 7.984 0.004907 **
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# same as the p value for the added parameter

lm_model_all <- lm(medv ~ ., data = df_boston)
# Call:
# lm(formula = medv ~ ., data = df_boston)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -15.595  -2.730  -0.518   1.777  26.199
#
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)
# (Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
# crim        -1.080e-01  3.286e-02  -3.287 0.001087 **
# zn           4.642e-02  1.373e-02   3.382 0.000778 ***
# indus        2.056e-02  6.150e-02   0.334 0.738288
# chas         2.687e+00  8.616e-01   3.118 0.001925 **
# nox         -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
# rm           3.810e+00  4.179e-01   9.116  < 2e-16 ***
# age          6.922e-04  1.321e-02   0.052 0.958229
# dis         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
# rad          3.060e-01  6.635e-02   4.613 5.07e-06 ***
# tax         -1.233e-02  3.760e-03  -3.280 0.001112 **
# ptratio     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
# black        9.312e-03  2.686e-03   3.467 0.000573 ***
# lstat       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.745 on 492 degrees of freedom
# Multiple R-squared:  0.7406,    Adjusted R-squared:  0.7338
# F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16

car::vif(lm_model_all)

# crim       zn    indus     chas      nox       rm      age      dis
# 1.792192 2.298758 3.991596 1.073995 4.393720 1.933744 3.100826 3.955945
#  rad      tax  ptratio    black    lstat
# 7.484496 9.008554 1.799084 1.348521 2.941491

# All reasonably low


# All but a certain variable
lm_model_all_but_age <- lm(medv ~. -age, data = df_boston)


print(summary(lm_model_all_but_age))
# Call:
# lm(formula = medv ~ . - age, data = df_boston)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -15.6054  -2.7313  -0.5188   1.7601  26.2243
#
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)
# (Intercept)  36.436927   5.080119   7.172 2.72e-12 ***
# crim         -0.108006   0.032832  -3.290 0.001075 **
# zn            0.046334   0.013613   3.404 0.000719 ***
# indus         0.020562   0.061433   0.335 0.737989
# chas          2.689026   0.859598   3.128 0.001863 **
# nox         -17.713540   3.679308  -4.814 1.97e-06 ***
# rm            3.814394   0.408480   9.338  < 2e-16 ***
# dis          -1.478612   0.190611  -7.757 5.03e-14 ***
# rad           0.305786   0.066089   4.627 4.75e-06 ***
# tax          -0.012329   0.003755  -3.283 0.001099 **
# ptratio      -0.952211   0.130294  -7.308 1.10e-12 ***
# black         0.009321   0.002678   3.481 0.000544 ***
# lstat        -0.523852   0.047625 -10.999  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.74 on 493 degrees of freedom
# Multiple R-squared:  0.7406,    Adjusted R-squared:  0.7343
# F-statistic: 117.3 on 12 and 493 DF,  p-value: < 2.2e-16

# Note age is not present now

# Interactions

print(summary(lm(medv ~ lstat * age, data=df_boston)))
Call:
lm(formula = medv ~ lstat * age, data = df_boston)

# Residuals:
#     Min      1Q  Median      3Q     Max
# -15.806  -4.045  -1.333   2.085  27.552
#
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)
# (Intercept) 36.0885359  1.4698355  24.553  < 2e-16 ***
# lstat       -1.3921168  0.1674555  -8.313 8.78e-16 ***
# age         -0.0007209  0.0198792  -0.036   0.9711
# lstat:age    0.0041560  0.0018518   2.244   0.0252 *
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 6.149 on 502 degrees of freedom
# Multiple R-squared:  0.5557,    Adjusted R-squared:  0.5531
# F-statistic: 209.3 on 3 and 502 DF,  p-value: < 2.2e-16


lm_fit_linear <- lm(medv ~ lstat, data = df_boston)
lm_fit_quadratic <- lm(medv ~ lstat + I(lstat ^ 2), data = df_boston)

print(anova(lm_fit_linear, lm_fit_quadratic))
# Model 1: medv ~ lstat
# Model 2: medv ~ lstat + I(lstat^2)
#   Res.Df   RSS Df Sum of Sq     F    Pr(>F)
# 1    504 19472
# 2    503 15347  1    4125.1 135.2 < 2.2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# Same as the squared t stat for the quadratic term

preds_df <- data.frame(
  predicted_value = c(predict(lm_fit_linear), predict(lm_fit_quadratic)),
  studentized_residual = c(residuals(lm_fit_linear), residuals(lm_fit_quadratic))
)
preds_df$model_type <- c(rep("linear", nrow(df_boston)), rep("quadratic", nrow(df_boston)))

residual_plot_comparison <- ggplot2::ggplot(data = preds_df) +
  ggplot2::geom_smooth(ggplot2::aes(x = predicted_value, y = studentized_residual, color = model_type))


# Include polynomials of arbitrary order
summary(lm(medv ~ poly(lstat, 5), data = df_boston))

# Call:
# lm(formula = medv ~ poly(lstat, 5), data = df_boston)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -13.5433  -3.1039  -0.7052   2.0844  27.1153
#
# Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)
# (Intercept)       22.5328     0.2318  97.197  < 2e-16 ***
# poly(lstat, 5)1 -152.4595     5.2148 -29.236  < 2e-16 ***
# poly(lstat, 5)2   64.2272     5.2148  12.316  < 2e-16 ***
# poly(lstat, 5)3  -27.0511     5.2148  -5.187 3.10e-07 ***
# poly(lstat, 5)4   25.4517     5.2148   4.881 1.42e-06 ***
# poly(lstat, 5)5  -19.2524     5.2148  -3.692 0.000247 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 5.215 on 500 degrees of freedom
# Multiple R-squared:  0.6817,    Adjusted R-squared:  0.6785
# F-statistic: 214.2 on 5 and 500 DF,  p-value: < 2.2e-16

df_carseats <- Carseats

summary(lm(Sales ~ . + Income:Advertising + Price:Age, data = df_carseats))


contrasts(df_carseats$ShelveLoc)
# Good Medium
# Bad       0      0
# Good      1      0
# Medium    0      1
