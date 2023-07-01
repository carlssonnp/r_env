library(MASS)
library(ggplot2)
library(GGally)
library(dplyr)
library(gridExtra)
library(ISLR)

image_path <- "isl/chapter_3/images/"


df_auto <- Auto


simple_linear_regression_model <- lm(mpg ~ horsepower, df_auto)

print(summary(simple_linear_regression_model))
#
# Call:
# lm(formula = mpg ~ horsepower, data = df_auto)
#
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

# 1. Large F statistic / large t statistic in this case (simple linear regression)
# indicate that there is a relationship between the predictor and the response
# 2. R-squared is 60%, indicating 60% of the variance is explained by the
# model. alternatively, compute % average error

print(summary(simple_linear_regression_model)$sigma / mean(df_auto$mpg))
# [1] 0.2092371

# So our average error is 20%

# 3. The relationship is negative.
# 4.

test_df <- data.frame(
  horsepower = 98
)

predict(simple_linear_regression_model, test_df, interval="confidence")
# fit      lwr      upr
# 1 24.46708 23.97308 24.96108
predict(simple_linear_regression_model, test_df, interval="prediction")
# fit     lwr      upr
# 1 24.46708 14.8094 34.12476

predictions_df <- data.frame(
  preds = predict(simple_linear_regression_model),
  mpg = df_auto$mpg,
  horsepower = df_auto$horsepower,
  studentized_residual = rstudent(simple_linear_regression_model)
)

regression_plot <- ggplot2::ggplot(data = predictions_df) +
  ggplot2::geom_point(ggplot2::aes(x = horsepower, y = mpg)) +
  ggplot2::geom_line(ggplot2::aes(x = horsepower, y = preds))

# evidence of non-linearity
ggplot2::ggsave(paste0(image_path, "/df_auto_simple_linear_regression_plot.png"), regression_plot)

diagnostic_plot <- ggplot2::ggplot(data = predictions_df) +
  ggplot2::geom_point(ggplot2::aes(x = preds, y = studentized_residual))
# again, evidence of non-linearity

ggplot2::ggsave(paste0(image_path, "/df_auto_simple_linear_regression_plot_errors.png"), diagnostic_plot)

# can plot diagnostics using base R

png(paste0(image_path, "/diagnostic_plot.png"))
par(mfrow = c(2,2))
plot(simple_linear_regression_model)
dev.off()

# Shows:
# 1. residuals vs fitted value
# 2. QQ plot
# 3. Scale location (sqrt(studentized_residual) vs fitted values)
# 4. studentized residuals vs leverage

# there aren't any points with high leverage that are also marked outliers,
# so the cooks distance is small for all observations. However, there is a noticeable trend
# in the plots of residuals vs fitted values, indicating a non-linear relationship.
# we see this in the qq plot as well

pairwise_plot <- GGally::ggpairs(data = df_auto %>% dplyr::select(., -name))
ggplot2::ggsave(paste0(image_path, "/auto_pairwise_plot.png"), pairwise_plot)

auto_lm_model <- lm(mpg ~ . - name, data = df_auto)


print(summary(auto_lm_model))

# Call:
# lm(formula = mpg ~ . - name, data = df_auto)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -9.5903 -2.1565 -0.1169  1.8690 13.0604
#
# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)
# (Intercept)  -17.218435   4.644294  -3.707  0.00024 ***
# cylinders     -0.493376   0.323282  -1.526  0.12780
# displacement   0.019896   0.007515   2.647  0.00844 **
# horsepower    -0.016951   0.013787  -1.230  0.21963
# weight        -0.006474   0.000652  -9.929  < 2e-16 ***
# acceleration   0.080576   0.098845   0.815  0.41548
# year           0.750773   0.050973  14.729  < 2e-16 ***
# origin         1.426141   0.278136   5.127 4.67e-07 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 3.328 on 384 degrees of freedom
# Multiple R-squared:  0.8215,    Adjusted R-squared:  0.8182
# F-statistic: 252.4 on 7 and 384 DF,  p-value: < 2.2e-16


# 1. Yes there is a relationship between the predictors and the response.
# 2. In the multiple regression, horsepower is not significant! Likely due to
# collinearity with other variables.
# 3. Newer cars have higher mpg.



png(paste0(image_path, "/diagnostic_plot_multiple.png"))
par(mfrow = c(2,2))
plot(auto_lm_model)
dev.off()


# definitely still a pattern to the residuals. observation 14 has high leverage but
# is not an outlier, so it doesn't affect the fit that much. There are also some
# outliers , but they don't have high leverage, so again they don't affect the
# fit that much.


lm_interactions <- lm(mpg~cylinders*displacement+displacement*weight, data = df_auto)
print(summary(lm_interactions))

# Call:
# lm(formula = mpg ~ cylinders * displacement + displacement *
#     weight, data = df_auto)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -13.2934  -2.5184  -0.3476   1.8399  17.7723
#
# Coefficients:
#                          Estimate Std. Error t value Pr(>|t|)
# (Intercept)             5.262e+01  2.237e+00  23.519  < 2e-16 ***
# cylinders               7.606e-01  7.669e-01   0.992    0.322
# displacement           -7.351e-02  1.669e-02  -4.403 1.38e-05 ***
# weight                 -9.888e-03  1.329e-03  -7.438 6.69e-13 ***
# cylinders:displacement -2.986e-03  3.426e-03  -0.872    0.384
# displacement:weight     2.128e-05  5.002e-06   4.254 2.64e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.103 on 386 degrees of freedom
# Multiple R-squared:  0.7272,    Adjusted R-squared:  0.7237
# F-statistic: 205.8 on 5 and 386 DF,  p-value: < 2.2e-16

# displacement and weight have a significant interaction effect



lm_horsepower <- lm(mpg ~ horsepower, data = df_auto)

print(summary(lm_horsepower))

# Call:
# lm(formula = mpg ~ horsepower, data = df_auto)
#
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

lm_horsepower_transformed <- lm(mpg ~ poly(horsepower, 2), data = df_auto)

print(summary(lm_horsepower_transformed))

# Call:
# lm(formula = mpg ~ poly(horsepower, 2), data = df_auto)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -14.7135  -2.5943  -0.0859   2.2868  15.8961
#
# Coefficients:
#                       Estimate Std. Error t value Pr(>|t|)
# (Intercept)            23.4459     0.2209  106.13   <2e-16 ***
# poly(horsepower, 2)1 -120.1377     4.3739  -27.47   <2e-16 ***
# poly(horsepower, 2)2   44.0895     4.3739   10.08   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.374 on 389 degrees of freedom
# Multiple R-squared:  0.6876,    Adjusted R-squared:  0.686
# F-statistic:   428 on 2 and 389 DF,  p-value: < 2.2e-16


anova(lm_horsepower, lm_horsepower_transformed)

# Analysis of Variance Table
#
# Model 1: mpg ~ horsepower
# Model 2: mpg ~ poly(horsepower, 2)
#   Res.Df    RSS Df Sum of Sq      F    Pr(>F)
# 1    390 9385.9
# 2    389 7442.0  1    1943.9 101.61 < 2.2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# (101.61  = 10.08^2)



# Carseats dataset
df_carseats <- Carseats

lm_carseats_full <- lm(Sales ~ Price + Urban + US, data = df_carseats)

print(summary(lm_carseats_full))

# Call:
# lm(formula = Sales ~ Price + Urban + US, data = df_carseats)

# Residuals:
#     Min      1Q  Median      3Q     Max
# -6.9206 -1.6220 -0.0564  1.5786  7.0581
#
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)
# (Intercept) 13.043469   0.651012  20.036  < 2e-16 ***
# Price       -0.054459   0.005242 -10.389  < 2e-16 ***
# UrbanYes    -0.021916   0.271650  -0.081    0.936
# USYes        1.200573   0.259042   4.635 4.86e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 2.472 on 396 degrees of freedom
# Multiple R-squared:  0.2393,    Adjusted R-squared:  0.2335
# F-statistic: 41.52 on 3 and 396 DF,  p-value: < 2.2e-16

# Higher price leads to lower sales
# urbanicity does not seem to affect sales
# whether or not it was manufactured in the US affects sales: being
# manufactured in the US increases the sales

lm_carseats_small <- lm(Sales ~ Price + US, data = df_carseats)


print(summary(lm_carseats_small))
#
# Call:
# lm(formula = Sales ~ Price + US, data = df_carseats)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -6.9269 -1.6286 -0.0574  1.5766  7.0515
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 13.03079    0.63098  20.652  < 2e-16 ***
# Price       -0.05448    0.00523 -10.416  < 2e-16 ***
# USYes        1.19964    0.25846   4.641 4.71e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 2.469 on 397 degrees of freedom
# Multiple R-squared:  0.2393,    Adjusted R-squared:  0.2354
# F-statistic: 62.43 on 2 and 397 DF,  p-value: < 2.2e-16


# We see they fit the training data equally well; in fact the adjusted r squared
# is slightly higher for the smaller model. the average percent error
# is

print(summary(lm_carseats_small)$sigma / abs(mean(df_carseats$Sales)))
# [1] 0.3294143

# Note that RSS always decreases with more variables, as does r squared, but the RSE
# could increase with more variables since you are dividing by a smaller number (n-p)


confint(lm_carseats_small)

# 2.5 %      97.5 %
# (Intercept) 11.79032020 14.27126531
# Price       -0.06475984 -0.04419543
# USYes        0.69151957  1.70776632


png(paste0(image_path, "/diagnostic_plot_carseats.png"))
par(mfrow = c(2,2))
plot(lm_carseats_small)
dev.off()

# There are some points with high leverage, but they aren't outliers. All studentized
# residuals are within 3 standard deviations of the mean (0), so we don't see any outliers.


# 11
set.seed(1)
x <- rnorm(100)
y <- 2 * x + rnorm(100)

print(summary(lm(y~x + 0)))
# Call:
# lm(formula = y ~ x + 0)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -1.9154 -0.6472 -0.1771  0.5056  2.3109
#
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)
# x   1.9939     0.1065   18.73   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.9586 on 99 degrees of freedom
# Multiple R-squared:  0.7798,    Adjusted R-squared:  0.7776
# F-statistic: 350.7 on 1 and 99 DF,  p-value: < 2.2e-16


print(summary(lm(x~y + 0)))
# Call:
# lm(formula = x ~ y + 0)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -0.8699 -0.2368  0.1030  0.2858  0.8938
#
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)
# y  0.39111    0.02089   18.73   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4246 on 99 degrees of freedom
# Multiple R-squared:  0.7798,    Adjusted R-squared:  0.7776
# F-statistic: 350.7 on 1 and 99 DF,  p-value: < 2.2e-16


# The F statistic is the msame, as is the r squared, and the t value of
# predictor.


print(summary(lm(y~x)))
# Call:
# lm(formula = y ~ x)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -1.8768 -0.6138 -0.1395  0.5394  2.3462
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) -0.03769    0.09699  -0.389    0.698
# x            1.99894    0.10773  18.556   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.9628 on 98 degrees of freedom
# Multiple R-squared:  0.7784,    Adjusted R-squared:  0.7762
# F-statistic: 344.3 on 1 and 98 DF,  p-value: < 2.2e-16

print(summary(lm(x~y)))
# Call:
# lm(formula = x ~ y)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -0.90848 -0.28101  0.06274  0.24570  0.85736
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)  0.03880    0.04266    0.91    0.365
# y            0.38942    0.02099   18.56   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.4249 on 98 degrees of freedom
# Multiple R-squared:  0.7784,    Adjusted R-squared:  0.7762
# F-statistic: 344.3 on 1 and 98 DF,  p-value: < 2.2e-16


x <- rnorm(100)
y <- x

summary(lm(y ~ x + 0))

# Call:
# lm(formula = y ~ x + 0)
#
# Residuals:
#        Min         1Q     Median         3Q        Max
# -1.100e-16 -1.490e-17  2.900e-18  2.820e-17  3.934e-15
#
# Coefficients:
#    Estimate Std. Error   t value Pr(>|t|)
# x 1.000e+00  4.013e-17 2.492e+16   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 3.984e-16 on 99 degrees of freedom
# Multiple R-squared:      1,     Adjusted R-squared:      1
# F-statistic: 6.211e+32 on 1 and 99 DF,  p-value: < 2.2e-16
#
# Warning message:
# In summary.lm(lm(y ~ x + 0)) :
#   essentially perfect fit: summary may be unreliable

summary(lm(x ~ y + 0))
#
# Call:
# lm(formula = x ~ y + 0)
#
# Residuals:
#        Min         1Q     Median         3Q        Max
# -1.100e-16 -1.490e-17  2.900e-18  2.820e-17  3.934e-15
#
# Coefficients:
#    Estimate Std. Error   t value Pr(>|t|)
# y 1.000e+00  4.013e-17 2.492e+16   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 3.984e-16 on 99 degrees of freedom
# Multiple R-squared:      1,     Adjusted R-squared:      1
# F-statistic: 6.211e+32 on 1 and 99 DF,  p-value: < 2.2e-16


# If we add any noise to y, the coefficients will not be the same, or if we
# make the relationship different by a scalar


# 13.

generate_least_squares_plot <- function(var_eps) {

  x <- rnorm(100)
  eps <- rnorm(100, 0, var_eps)
  y_true <- -1 + 0.5 * x
  y_observed <- y_true + eps

  lm_model <- lm(y_observed~x)

  print(summary(lm_model))


  print(confint(lm_model))

  y_pred <- predict(lm_model)

  df <- data.frame(
    x = rep(x, 3),
    y = c(y_true, y_observed, y_pred),
    response_type = c(
      rep("true_value", length(y_pred)), rep("observed_value", length(y_pred)),
      rep("predicted_value", length(y_pred))
    )
  )

  print(summary(lm(y_observed ~ poly(x, 2))))

  ggplot2::ggplot(data = df %>% dplyr::filter(., response_type == "observed_value")) +
    ggplot2::geom_point(ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line(data = df %>% dplyr::filter(., response_type != "observed_value"), ggplot2::aes(x = x, y = y, color = response_type)) +
    ggplot2::labs(title = paste("Eps variance: ", var_eps))

}
set.seed(1)

EPS_VARS <- c(0.00000000001, 0.25, 1) %>%
  stats::setNames(., .)

plots <- lapply(EPS_VARS, generate_least_squares_plot)


all_plots <- do.call(gridExtra::grid.arrange, plots)

# With smaller variance of the error term, we obtain better estimates of the
# coefficients (smaller standard errors)


# As we increase the variance of the error term, the polynomial fit fits the
# training data better (using r-squared, which always goes up, or RSE, which goes down
# comparing linear to polynomial only in the noisiest case)


ggplot2::ggsave(paste0(image_path, "/variance_experiments.png"), all_plots)



# Problem 14
set.seed(1)
x1 <- runif(100)
x2 <- 0.5 * x1 + rnorm(100) / 10
y <- 2 + 2 * x1 + 0.3 * x2 + rnorm(100)

cor(x1, x2)
# [1] 0.8351212

lm_model <- lm(y ~ x1 + x2)

print(summary(lm_model))

# Call:
# lm(formula = y ~ x1 + x2)

# Residuals:
#     Min      1Q  Median      3Q     Max
# -2.8311 -0.7273 -0.0537  0.6338  2.3359
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.1305     0.2319   9.188 7.61e-15 ***
# x1            1.4396     0.7212   1.996   0.0487 *
# x2            1.0097     1.1337   0.891   0.3754
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.056 on 97 degrees of freedom
# Multiple R-squared:  0.2088,    Adjusted R-squared:  0.1925
# F-statistic:  12.8 on 2 and 97 DF,  p-value: 1.164e-05


# x2 is not significant, and x1 is barely significant
# intercept estimate is close to population value, but x1 and x2 are far from the
# true values


print(summary(lm(y ~ x1)))

# Call:
# lm(formula = y ~ x1)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -2.89495 -0.66874 -0.07785  0.59221  2.45560
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.1124     0.2307   9.155 8.27e-15 ***
# x1            1.9759     0.3963   4.986 2.66e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.055 on 98 degrees of freedom
# Multiple R-squared:  0.2024,    Adjusted R-squared:  0.1942
# F-statistic: 24.86 on 1 and 98 DF,  p-value: 2.661e-06

print(summary(lm(y ~ x2)))
# Call:
# lm(formula = y ~ x2)

# Residuals:
#      Min       1Q   Median       3Q      Max
# -2.62687 -0.75156 -0.03598  0.72383  2.44890
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.3899     0.1949   12.26  < 2e-16 ***
# x2            2.8996     0.6330    4.58 1.37e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.072 on 98 degrees of freedom
# Multiple R-squared:  0.1763,    Adjusted R-squared:  0.1679
# F-statistic: 20.98 on 1 and 98 DF,  p-value: 1.366e-05


# x2 looks very significant now, although the estimate is far from the true value
# basically the effect of x1 is being included in x2 since they are so highly correlated


# No contradiction here; the high correlation leads to high variance estimates
# for the coefficients when they are in the model, meaning the estimates could be far
# from their population counterparts and the p values might be high.


x1 <- c(x1, 0.1)
x2 <- c(x2, 0.8)
y <- c(y, 6)

lm_model_full <- lm(y ~ x1 + x2)

print(summary(lm_model))
Call:
# lm(formula = y ~ x1 + x2)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -2.73348 -0.69318 -0.05263  0.66385  2.30619
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.2267     0.2314   9.624 7.91e-16 ***
# x1            0.5394     0.5922   0.911  0.36458
# x2            2.5146     0.8977   2.801  0.00614 **
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.075 on 98 degrees of freedom
# Multiple R-squared:  0.2188,    Adjusted R-squared:  0.2029
# F-statistic: 13.72 on 2 and 98 DF,  p-value: 5.564e-06

lm_model_x1 <- lm(y ~ x1)

print(summary(lm_model_x1))
# Call:
# lm(formula = y ~ x1)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -2.8897 -0.6556 -0.0909  0.5682  3.5665
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.2569     0.2390   9.445 1.78e-15 ***
# x1            1.7657     0.4124   4.282 4.29e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.111 on 99 degrees of freedom
# Multiple R-squared:  0.1562,    Adjusted R-squared:  0.1477
# F-statistic: 18.33 on 1 and 99 DF,  p-value: 4.295e-05

lm_model_x2 <- lm(y ~ x2)

print(summary(lm_model_x2))
# Call:
# lm(formula = y ~ x2)
#
# Residuals:
#      Min       1Q   Median       3Q      Max
# -2.64729 -0.71021 -0.06899  0.72699  2.38074
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)   2.3451     0.1912  12.264  < 2e-16 ***
# x2            3.1190     0.6040   5.164 1.25e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.074 on 99 degrees of freedom
# Multiple R-squared:  0.2122,    Adjusted R-squared:  0.2042
# F-statistic: 26.66 on 1 and 99 DF,  p-value: 1.253e-06



# Now the estimates are even further off from the population values


png(paste0(image_path, "/diagnostic_plot_full_exercise_14.png"))
par(mfrow = c(2,2))
plot(lm_model_full)
dev.off()

png(paste0(image_path, "/diagnostic_plot_x1_exercise_14.png"))
par(mfrow = c(2,2))
plot(lm_model_x1)
dev.off()

png(paste0(image_path, "/diagnostic_plot_x2_exercise_14.png"))
par(mfrow = c(2,2))
plot(lm_model_x2)
dev.off()


# In the model with both x1 and x2, the point is not technically an outlier, but
# has very high leverage and a relatively large residual, so it has an outsize influence on the fit


# In the model with x1 only, the observation has less leverage but the the residual
# is larger, so it also affects the fit

# in the model with x2 only, it doesn't have particularly large leverage and is not
# an outlier


df_boston <- Boston

models <- list()
for (nm in colnames(df_boston)) {
  models[[nm]] <- summary(lm(as.formula(paste("crim ~", nm)), data = df_boston))
}


model_coefficients <- lapply(models, function(model) coef(model))
print(model_coefficients)

# $crim
#             Estimate Std. Error  t value     Pr(>|t|)
# (Intercept) 3.613524  0.3823853 9.449954 1.262593e-19
#
# $zn
#                Estimate Std. Error   t value     Pr(>|t|)
# (Intercept)  4.45369376  0.4172178 10.674746 4.037668e-24
# zn          -0.07393498  0.0160946 -4.593776 5.506472e-06
#
# $indus
#               Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -2.0637426 0.66722830 -3.093008 2.091266e-03
# indus        0.5097763 0.05102433  9.990848 1.450349e-21
#
# $chas
#              Estimate Std. Error   t value     Pr(>|t|)
# (Intercept)  3.744447  0.3961111  9.453021 1.239505e-19
# chas        -1.892777  1.5061155 -1.256727 2.094345e-01
#
# $nox
#              Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -13.71988   1.699479 -8.072992 5.076814e-15
# nox          31.24853   2.999190 10.418989 3.751739e-23
#
# $rm
#              Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) 20.481804  3.3644742  6.087669 2.272000e-09
# rm          -2.684051  0.5320411 -5.044819 6.346703e-07
#
# $age
#               Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -3.7779063 0.94398472 -4.002084 7.221718e-05
# age          0.1077862 0.01273644  8.462825 2.854869e-16
#
# $dis
#              Estimate Std. Error   t value     Pr(>|t|)
# (Intercept)  9.499262  0.7303972 13.005611 1.502748e-33
# dis         -1.550902  0.1683300 -9.213458 8.519949e-19
#
# $rad
#               Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -2.2871594 0.44347583 -5.157349 3.605846e-07
# rad          0.6179109 0.03433182 17.998199 2.693844e-56
#
# $tax
#                Estimate  Std. Error   t value     Pr(>|t|)
# (Intercept) -8.52836909 0.815809392 -10.45387 2.773600e-23
# tax          0.02974225 0.001847415  16.09939 2.357127e-47
#
# $ptratio
#               Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -17.646933  3.1472718 -5.607057 3.395255e-08
# ptratio       1.151983  0.1693736  6.801430 2.942922e-11
#
# $black
#                Estimate  Std. Error   t value     Pr(>|t|)
# (Intercept) 16.55352922 1.425902755 11.609157 8.922239e-28
# black       -0.03627964 0.003873154 -9.366951 2.487274e-19
#
# $lstat
#               Estimate Std. Error   t value     Pr(>|t|)
# (Intercept) -3.3305381 0.69375829 -4.800718 2.087022e-06
# lstat        0.5488048 0.04776097 11.490654 2.654277e-27
#
# $medv
#               Estimate Std. Error  t value     Pr(>|t|)
# (Intercept) 11.7965358 0.93418916 12.62757 5.934119e-32
# medv        -0.3631599 0.03839017 -9.45971 1.173987e-19


# Everything other than chas is significant

univariate_coefficients <- lapply(
  model_coefficients[-1],
  function(coefficients) {
    coefficients[2, 1]
  }
)


lm_full <- summary(lm(crim ~ ., data = df_boston))

print(lm_full)


# Call:
# lm(formula = crim ~ ., data = df_boston)
#
# Residuals:
#    Min     1Q Median     3Q    Max
# -9.924 -2.120 -0.353  1.019 75.051
#
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)
# (Intercept)  17.033228   7.234903   2.354 0.018949 *
# zn            0.044855   0.018734   2.394 0.017025 *
# indus        -0.063855   0.083407  -0.766 0.444294
# chas         -0.749134   1.180147  -0.635 0.525867
# nox         -10.313535   5.275536  -1.955 0.051152 .
# rm            0.430131   0.612830   0.702 0.483089
# age           0.001452   0.017925   0.081 0.935488
# dis          -0.987176   0.281817  -3.503 0.000502 ***
# rad           0.588209   0.088049   6.680 6.46e-11 ***
# tax          -0.003780   0.005156  -0.733 0.463793
# ptratio      -0.271081   0.186450  -1.454 0.146611
# black        -0.007538   0.003673  -2.052 0.040702 *
# lstat         0.126211   0.075725   1.667 0.096208 .
# medv         -0.198887   0.060516  -3.287 0.001087 **
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 6.439 on 492 degrees of freedom
# Multiple R-squared:  0.454,     Adjusted R-squared:  0.4396
# F-statistic: 31.47 on 13 and 492 DF,  p-value: < 2.2e-16


full_coefficients <- coef(lm_full)

coefficient_pairings <- list()
for (nm in names(univariate_coefficients)) {
  coefficient_pairings[[nm]] <- c(univariate = univariate_coefficients[[nm]], multivariate = full_coefficients[nm, "Estimate"])
}

df_for_plotting <- data.frame(var = names(coefficient_pairings))
df_for_plotting$univariate <- sapply(coefficient_pairings, function(lst) lst["univariate"])
df_for_plotting$multivariate <- sapply(coefficient_pairings, function(lst) lst["multivariate"])

plt <- ggplot2::ggplot(data = df_for_plotting) +
  ggplot2::geom_point(ggplot2::aes(x = univariate, y = multivariate)) +
  ggplot2::geom_text(ggplot2::aes(x = univariate, y = multivariate, label = var))



lm_linear <- lm(crim ~ medv, data = df_boston)
print(summary(lm_linear))

# Call:
# lm(formula = crim ~ medv, data = df_boston)
#
# Residuals:
#    Min     1Q Median     3Q    Max
# -9.071 -4.022 -2.343  1.298 80.957
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 11.79654    0.93419   12.63   <2e-16 ***
# medv        -0.36316    0.03839   -9.46   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 7.934 on 504 degrees of freedom
# Multiple R-squared:  0.1508,    Adjusted R-squared:  0.1491
# F-statistic: 89.49 on 1 and 504 DF,  p-value: < 2.2e-16

lm_poly <- lm(crim ~ poly(medv, 5), data = df_boston)
print(summary(lm_poly))
# Call:
# lm(formula = crim ~ poly(medv, 3), data = df_boston)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -24.427  -1.976  -0.437   0.439  73.655
#
# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)
# (Intercept)       3.614      0.292  12.374  < 2e-16 ***
# poly(medv, 3)1  -75.058      6.569 -11.426  < 2e-16 ***
# poly(medv, 3)2   88.086      6.569  13.409  < 2e-16 ***
# poly(medv, 3)3  -48.033      6.569  -7.312 1.05e-12 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 6.569 on 502 degrees of freedom
# Multiple R-squared:  0.4202,    Adjusted R-squared:  0.4167
# F-statistic: 121.3 on 3 and 502 DF,  p-value: < 2.2e-16

# The polynomial fit explains much more variance!

ggplot(data = df_boston) +
  geom_point(aes(x = medv, y = crim))

png(paste0(image_path, "/diagnostic_plot_lm_linear_exercise_15.png"))
par(mfrow = c(2,2))
plot(lm_linear)
dev.off()

png(paste0(image_path, "/diagnostic_plot_lm_poly_exercise_15.png"))
par(mfrow = c(2,2))
plot(lm_poly)
dev.off()

n <- 10000
reps <- 100
x <- rnorm(n)
res <- c()
for (i in seq(reps)) {
  y <- x + runif(n, -100, 100)
  res[i] <- coef(summary(lm(y ~ x)))["x", "Estimate"]
}


df <- data.frame(coefficient = res)

ggplot(df) + geom_histogram(aes(x=coefficient), bins=15)
