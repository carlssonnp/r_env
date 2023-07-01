library(e1071)
library(ROCR)


set.seed(1)

x <- matrix(rnorm(20 * 2), 20, 2)
y <- c(rep(-1, 10), rep(1, 10))

x[y == 1, ]<- x[y == 1, ] + 1

plot(x, col=3 - y)


# Not linearly separable


df <- data.frame(x = x, y = as.factor(y))


model <- svm(y ~ ., data = df, kernel = "linear", cost = 10)


plot(model, df)

model <- svm(y ~ ., data = df, kernel = "linear", cost = 0.1)

cv_model <- tune(svm, y ~ ., data = df, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

model <- cv_model$best.model

x_test <- matrix(rnorm(20 * 2), 20, 2)
y_test <- c(rep(-1, 10), rep(1, 10))

x_test[y_test == 1, ] <- x_test[y_test == 1, ] + 1

df_test <- data.frame(x = x_test, y = y_test)

preds <- predict(model, df_test)

table(preds, df_test$y)

# 3 misclassifications

x[y == 1, ] <- x[y == 1] + 0.5

df <- data.frame(x = x, y = as.factor(y))

# Maximal margin hyperplane (margin is very small)
model <- svm(y ~ ., data = df, cost = 1e5, kernel = "linear")


plot(model, df)


x <- matrix(rnorm(200 * 2), ncol = 2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150] <- x[101:150] - 2
y <- c(rep(1, 150), rep(2, 50))


df <- data.frame(x = x, y = as.factor(y))

train_idx <- sample(200, 100)

df_train <- df[train_idx, ]
df_test <- df[-train_idx, ]

model <- svm(y ~ ., data = df_train, kernel = "radial", gamma = 1, cost = 1)

plot(model, df_train)

model <- svm(y ~ ., data = df_train, kernel = "radial", gamma = 1, cost = 1e5)

plot(model, df_train)

tuned_model <- tune(svm, y ~ ., data = df_train, kernel = "radial", ranges = list(cost = c(0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3, 4)))


preds <- predict(tuned_model$best.model, df_test)

table(preds, df_test$y)


rocplot <- function(pred, truth, ...) {
  predob <- prediction(pred, truth)
  plot(performance(predob, "tpr", "fpr"))
}


model <- svm(y ~ ., data = df_train, kernel = "radial", gamma = 2, cost = 1, decision.values = TRUE)


fitted_values <- attributes(predict(model, df_train, decision.values = TRUE))$decision.values

rocplot(fitted_values, df_train$y)


x <- rbind(x, matrix(rnorm(50 * 2), ncol = 2))

y <- c(y, rep(0, 50))

# One vs one
df <- data.frame(x = x, y = as.factor(y))

model <- svm(y ~ ., data = df, kernel = "radial", cost = 10, gamma = 1)

plot(model, df)

df <- data.frame(x = Khan$xtrain, y = as.factor(Khan$ytrain))

model <- svm(y ~ ., data = df, kernel = "linear", cost = 10)


table(model$fitted, df$y)

# No training errors due to high dimensionality of data

df_test <- data.frame(x = Khan$xtest, y = as.factor(Khan$ytest))

preds <- predict(model, df_test)

table(preds, df_test$y)


# 2 errors



# Applied

x <- matrix(rnorm(200 * 2), 200, 2)
y <- c(rep(0, 100), rep(1, 100))
x[1:50, ] <- x[1:50, ] + 2
x[51:100, ] <- x[51:100, ] - 2

plot(x[, 1], x[, 2])

df <- data.frame(x = x, y = as.factor(y))

train_idx <- sample(nrow(df), nrow(df) %/% 2)

df_train <- df[train_idx, ]
df_test <- df[-train_idx, ]


ggplot(data = df_train) +
  geom_point(aes(x = x.1, y = x.2, color = y))


linear_model <- svm(y ~ ., data = df_train, kernel = "linear", cost = 10)

plot(linear_model, df_train)

# Terrible

radial_model <- svm(y ~ ., data = df_train, kernel = "radial", cost = 10)

# much better
polynomial_model <- svm(y ~ ., data = df_train, kernel = "polynomial", cost = 10, degree = 2)

 # Also better


preds_polynomial <- predict(polynomial_model, df_test)

table(preds_polynomial, df_test$y)

# Good

table(predict(radial_model, df_test), df_test$y)

table(predict(linear_model, df_test), df_test$y)

# predicts everything to be 1


x1 <- runif(500) - 0.5
x2 <- runif(500) - 0.5

y <- 1 * ((x1 ^ 2 - x2 ^ 2) > 0 )

plot(x1, x2, col = 3 - y)
# looks pretty non-linear


logistic_regression_model <- glm(y ~ x1 + x2, family = "binomial")


preds <- predict(logistic_regression_model, type = "response")

preds <- ifelse(preds >= 0.5, 1, 0)


plot(x1, x2, col = 3 - preds)


logistic_regression_model <- glm(y ~ I(x1 ^ 2) + I(x2 ^ 2), family = "binomial")

preds <- predict(logistic_regression_model, type = "response")

preds <- ifelse(preds >= 0.5, 1, 0)

plot(x1, x2, col = 3 - preds)


# Huge standard errors but good predictions

model <- svm(y ~ x1 + x2, kernel = "linear")

preds <- predict(model)

preds <- ifelse(preds >= 0.5, 1, 0)

plot(x1, x2, col = 3 - preds)

# Linear
model <- svm(y ~ x1 + x2, kernel = "radial")

preds <- predict(model)

preds <- ifelse(preds >= 0.5, 1, 0)

plot(x1, x2, col = 3 - preds)

set.seed(1)
x <- matrix(rnorm(200 * 2), 200, 2)
x[1:100] <- x[1:100] + 5
y <- c(rep(0, 100), rep(1, 100))


plot(x, col = 2 - y)


cv_models <- tune(svm, x, as.factor(y), kernel = "linear", ranges = list(cost = c(0.001, 0.1, 0.05, 1, 10, 100, 1e5)))

# Error is very high for low C and for high C

x <- matrix(rnorm(200 * 2), 200, 2)
x[1:100] <- x[1:100] + 5
y <- c(rep(0, 100), rep(1, 100))

preds <- predict(cv_models$best.model, x)


table(preds, y)

Auto$dep_var <- as.factor(ifelse(Auto$mpg >= median(Auto$mpg), 1, 0))

model <- svm(dep_var ~ . -mpg, kernel = "linear", data = Auto)

tuned_model <- tune(svm, dep_var ~ . -mpg, data = Auto, kernel = "linear", ranges = list(cost = c(0.001, 0.1, 0.05, 1, 10, 100, 1e5)))


summary(tuned_model)

# Parameter tuning of ‘svm’:
#
# - sampling method: 10-fold cross validation
#
# - best parameters:
#  cost
#     1
#
# - best performance: 0.09429487
#
# - Detailed performance results:
#    cost      error dispersion
# 1 1e-03 0.12743590 0.04137670
# 2 1e-01 0.09685897 0.03752960
# 3 5e-02 0.09685897 0.03752960
# 4 1e+00 0.09429487 0.04964301
# 5 1e+01 0.11198718 0.06341659
# 6 1e+02 0.12980769 0.06460293
# 7 1e+05 0.11980769 0.05635070


tuned_model <- tune(svm, dep_var ~ . -mpg, data = Auto, ranges = list(cost = c(0.001, 0.1, 0.05, 1, 10, 100, 1e5), gamma = c(0.01, 0.1, 1), degree = c(3, 4, 5), kernel = c("polynomial", "radial")))


# radial kernel had best performance

train_idx <- sample(nrow(OJ), 800)

df_train <- OJ[train_idx, ]
df_test <- OJ[-train_idx, ]

model <- svm(Purchase ~ ., data = df_train, kernel = "linear", cost = 0.01)


preds_train <- predict(model)

table(preds_train, df_train$Purchase)

preds_test <- predict(model, df_test)

table(preds_test, df_test$Purchase)
# preds_test  CH  MM
#         CH 140  25
#         MM  21  84


tuned_cv <- tune(svm, Purchase ~ ., data = df_train,  kernel = "linear", ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))


tuned_cv
# Parameter tuning of ‘svm’:
#
# - sampling method: 10-fold cross validation
#
# - best parameters:
#  cost
#     5
#
# - best performance: 0.1625

tuned_cv <- tune(svm, Purchase ~ ., data = df_train,  kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))


# Parameter tuning of ‘svm’:
#
# - sampling method: 10-fold cross validation
#
# - best parameters:
#  cost
#     1
#
# - best performance: 0.1675



tuned_cv <- tune(svm, Purchase ~ ., data = df_train,  kernel = "polynomial", degree = 2, ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))


# all are pretty similar
