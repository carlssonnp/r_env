library(ISLR)
library(ggplot2)
library(glmnet)
library(leaps)


# Ridge
y <- 10
lambda <- 1

beta <- 1:100

loss <- (y - beta) ^ 2 + lambda * beta ^2

df <- data.frame(beta = beta, loss = loss)

ggplot(data = df) +
  geom_point(aes(x = beta, y = loss))


# The minimum occurs at beta = 5; agrees with

# 10 / (1 + 1)


# Lasso

y <- 10
lambda <- 1
beta <- 1:100

loss <- (y - beta) ^ 2 + lambda * abs(beta)

df <- data.frame(beta = beta, loss = loss)

ggplot(data = df) +
  geom_point(aes(x = beta, y = loss))


which.min(loss)

# [1] 9

# which agrees with y - lambda / 2 = 9.5

x <- rnorm(100)
eps <- rnorm(100, sd = 0.5)


y <- 10 + x + 2 * x  ^ 2 + 3 * x ^ 3 + eps


df <- data.frame(y = y, x = x)

for (i in seq(2, 10)) {
  df[paste("x", i, sep = "_")] <- df$x ^ i
}

# One way using complexity adjusted metrics, another way using cross-validation
best_subset <- regsubsets(y ~ ., data = df, nvmax = Inf)
plot(best_subset)

plot(best_subset, scale = "Cp")


plot(best_subset, scale = "adjr2")

# From the plots we see the correct 4 variable model has the lowest BIC and Cp, but
# adjusted r squared is higher for some slightly larger models

coef(best_subset, 3)
# (Intercept)           x         x_2         x_3
#   9.9152235   0.8323992   1.9412804   3.0450475


# Pretty close to the population regression line

forward_stepwise <- regsubsets(y ~ ., data = df, nvmax = Inf, method = "forward")

plot(forward_stepwise)

plot(forward_stepwise, scale = "Cp")

plot(forward_stepwise, scale = "adjr2")
# Again, going by adjusted r squared gives us the wrong answer

coef(forward_stepwise, 3)


# Exact same as best subset, since the varibles chosen are the same

backward_stepwise <- regsubsets(y ~ ., data = df, nvmax = Inf, method = "backward")

plot(backward_stepwise)

plot(backward_stepwise, scale = "Cp")

plot(backward_stepwise, scale = "adjr2")


# Same story
coef(backward_stepwise, 3)

# Same model chosen

# do it using cross-validation: two ways

# easier way, as shown in the chapter.

cross_validate <- function(df, k) {
  indices <- rep(seq(k), length.out = nrow(df))
  indices <- sample(indices)

  split(df, indices)
}



methods <- list(
  best_subset = best_subset, forward_stepwise = forward_stepwise,
  backward_stepwise = backward_stepwise
)

results <- vector("list", length = length(methods)) %>%
  setNames(., names(methods))

for (nm in names(methods)) {
  method <- methods[[nm]]
  mse_results <- matrix(0, nrow = 5, ncol = 10)
  for (i in seq(1:10)) {
    fields <- names(coef(method, i))
    fields <- setdiff(fields, "(Intercept)")
    fields <- paste(fields, collapse = " + ")
    equation <- as.formula(paste0("y ~ ", fields))

    folds <- cross_validate(df, 5)

    for (j in seq_along(folds)) {
      train <- do.call(rbind, folds[-j])
      test <- folds[[j]]
      model <- lm(equation, data = train)
      preds <- predict(model, test)
      mse <- mean((test$y - preds) ^ 2)
      mse_results[j, i] <- mse
    }
  }
  results[[nm]] <- apply(mse_results, 2, mean)
}

results

# $best_subset
#  [1] 5.4642674 0.5024521 0.3400414 0.3725265 0.3144843 0.3311377 0.3528499
#  [8] 8.7984348 4.2352847 0.7550205
#
# $forward_stepwise
#  [1] 5.8376883 0.4962530 0.3256897 0.3513884 0.3671411 0.3445627 0.3384756
#  [8] 0.4907862 1.2064426 1.1283362
#
# $backward_stepwise
#  [1] 5.5923454 0.5539730 0.3435039 0.3529398 0.5626703 0.3330510 0.3381713
#  [8] 0.4018133 0.7430248 1.0244744


# cross validation picks close to the correct model in each case


# harder way, as shown in the lab


k <- 5
methods <- c("exhaustive", "forward", "backward")


folds <- cross_validate(df, k)

results <- replicate(3, matrix(0, nrow = 5, ncol = 10), simplify = FALSE) %>%
  setNames(., methods)

for (i in seq(k)) {
  train <- do.call(rbind, folds[-i])
  test <- folds[[i]]

  for (method in methods) {
    model <- regsubsets(y ~ ., data = train, nvmax = Inf)

    for (j in seq(10)) {
      test_matrix <- model.matrix(y ~ ., data = test)
      coefs <- coef(model, id = j)
      test_matrix <- test_matrix[, names(coefs)]
      preds <- test_matrix %*% coefs

      results[[method]][i, j] <- mean((test$y - preds) ^ 2)
    }
  }
}

results <- lapply(results, function(mat) apply(mat, 2, mean))

# Picks the three variable model each time!


# Now do lasso

x_mat <- model.matrix(y ~., data = df)
x_mat <- x_mat[, -1]
y <- df$y

cv_glmnet <- cv.glmnet(x_mat, y, alpha = 1)

plot(cv_glmnet)

cv_glmnet$lambda.min
# [1] 0.03119191


# Now refit on entire dataset using lambda min

lasso_model <- glmnet::glmnet(x_mat, y, alpha = 1, lambda = cv_glmnet$lambda.1se)

coef(lasso_model)


# If we use 1se, the x^4 term gets dropped!


df$y <- 10 + 4 * df[["x_7"]] + eps


best_subset <- regsubsets(y ~ ., data = df)

plot(best_subset)

# model with intercept and x7 has highest BIC

# could also do this with cross-validation like before; it would tell us that the one variable model is best

x_mat <- model.matrix(y ~., data = df)
x_mat <- x_mat[, -1]


cv_glmnet <- cv.glmnet(x_mat, df$y, alpha = 1)

plot(cv_glmnet)

glmnet_model <- glmnet(x_mat, df$y, alpha = 1, lambda = cv_glmnet$lambda.min)
coef(glmnet_model)


# don't actually need to refit the model using  glmnet afterwards


# Exercise 8

library(ISLR)
set.seed(11)

train.size = dim(College)[1] / 2
train = sample(1:dim(College)[1], train.size)

df_college <- College
nrows <- nrow(df_college)

# train_idx <- sample(nrows, nrows %/% 2)

df_train <- df_college[train, ]
df_test <- df_college[-train, ]

y_train <- df_train$Apps
y_test <- df_test$Apps

x_train <- model.matrix(Apps ~. , data = df_train)
x_test <- model.matrix(Apps ~ ., data = df_test)


ols_model <- lm(Apps ~ ., data = df_train)

preds <- predict(ols_model, df_test)

sqrt(mean((df_test$Apps - preds) ^ 2))
# [1] 1295.546

# subset:
cross_validate <- function(df, k) {
  indices <- rep(seq(k), length.out = nrow(df))
  indices <- sample(indices)

  split(df, indices)
}

methods <- c("exhaustive", "forward", "backward")

models <- vector("list", length = length(methods)) %>% setNames(., methods)
results <- vector("list", length = length(methods)) %>% setNames(., methods)

for (method in methods) {
  models[[method]] <- leaps::regsubsets(Apps ~ ., data = df_train, method = method, nvmax = Inf)
  folds <- cross_validate(df_college, 5)
  rmse_results <- matrix(0, 5, 17)
  for (i in seq_along(folds)) {
    train <- do.call(rbind, folds[-i])
    train <- model.matrix(~ ., train) %>% as.data.frame() %>% select(., -"(Intercept)")
    test <- folds[[i]]
    test <- model.matrix(~ ., test) %>% as.data.frame()  %>% select(., -"(Intercept)")
    # browser()
    for (j in 1:17) {
      coefs <- names(coef(models[[method]], id = j))
      coefs <- setdiff(coefs, "(Intercept)")
      form <- as.formula(paste0("Apps ~", paste(coefs, collapse = " + ")))
      # browser()
      # browser()
      lm_model <- lm(form, data = train)
      preds <- predict(lm_model, test)

      rmse_results[i, j] <- sqrt(mean((test$Apps - preds) ^ 2))
    }
  }
  rmse_results <- apply(rmse_results, 2, mean)
  results[[method]] <- rmse_results
}

test_results <- results
n_vars <- lapply(results, which.min)

for (method in methods) {
  coefs <- coef(models[[method]], id = n_vars[[method]])
  predictions <- x_test[, names(coefs)] %*% coefs
  test_results[[method]] <- sqrt(mean((y_test - predictions) ^ 2))
}

# $exhaustive
# [1] 1155.754
#
# $forward
# [1] 1140.196
#
# $backward
# [1] 1138.035

# Quite similar results to using the full ols model


cv_ridge <- cv.glmnet(x_train[, -1], y_train, alpha = 0, thresh = 1e-17)
preds <- predict(cv_ridge, x_test[, -1], lambda = "lambda.min")

sqrt(mean((y_test - preds) ^ 2))
# [1] 1461.075

cv_lasso <- cv.glmnet(x_train[, -1], y_train, alpha = 1, thresh = 1e-17)
preds <- predict(cv_lasso, x_test[, -1], lambda = "lambda.min")

sqrt(mean((y_test - preds) ^ 2))

# [1] 1293.965



pcr_model <- pcr(Apps ~ ., data = df_train, validation = "CV")

validationplot(pcr_model)

preds <- predict(pcr_model, df_test, ncomp = 10)

sqrt(mean((y_test - preds) ^ 2))

# [1] 1024.906



pls_model <- plsr(Apps ~ ., data = df_train, validation = "CV")

preds <- predict(pls_model, df_test, ncomp = 10)

sqrt(mean((y_test - preds) ^ 2))

# [1] 1011.084


# Last exercise

X <- matrix(rnorm(20 * 1000), 1000, 20)
beta <- sample(5, 20, replace = TRUE)
zero_indices <- sample(20, 10)
beta[zero_indices] <- 0

y <- X %*% beta + rnorm(1000)

df <- data.frame(X)
df$Y <- y

nrows <- nrow(X)
train_idx <- sample(nrows, 100)

df_train <- df[train_idx, ]
df_test <- df[-train_idx, ]

best_subsets <- leaps::regsubsets(Y ~ ., data = df_train, nvmax = Inf)

test_results <- rep(0, 20)
train_results <- rep(0, 20)
x_test <- model.matrix(Y ~ ., df_test)
x_train <- model.matrix(Y ~ ., df_train)

for (i in seq(20)) {
  coefs <- coef(best_subsets, id = i)
  preds <- x_test[, names(coefs)] %*% coefs
  test_results[[i]] <- mean((df_test$Y - preds) ^ 2)

  preds <- x_train[, names(coefs)] %*% coefs
  train_results[[i]] <- mean((df_train$Y - preds) ^ 2)
}


df_results <- data.frame(n_vars = 1:20, mse_test = test_results, mse_train = train_results)

df_results

# n_vars   mse_test   mse_train
# 1       1 152.569433 148.6878991
# 2       2 124.557219 110.1117510
# 3       3  99.522346  83.3825273
# 4       4  75.644737  59.9054270
# 5       5  48.797685  39.3120638
# 6       6  31.615140  26.0752489
# 7       7  22.951060  15.7043949
# 8       8  12.657066   9.4526382
# 9       9   8.120637   6.5084703
# 10     10   7.347556   4.9644020
# 11     11   5.720264   3.9354313
# 12     12   4.840281   2.9538443
# 13     13   3.562249   2.2364757
# 14     14   2.242716   1.4144199
# 15     15   1.040955   0.7787844
# 16     16   1.056696   0.7570462
# 17     17   1.073879   0.7487939
# 18     18   1.076204   0.7455430
# 19     19   1.073712   0.7449128
# 20     20   1.076422   0.7445686

# Test error is minimized at the model including only the 15 features used in the
# population regression line. Train error just keeps decreasing.


# Extract coefficients

coef(best_subsets, id = 10)
# (Intercept)          X3          X5          X6         X10         X11
# -0.07271326  4.08233578  0.84963533  5.04929400  3.94214363  1.96051736
#         X13         X17         X18         X19         X20
#  4.00216056  1.92197098  4.91821257  0.96467036  1.90387232

beta
# [1] 0 0 4 0 1 5 0 0 0 4 2 0 4 0 0 0 2 5 1 2


# compares pretty nicely!

squared_coefficient_differences <- rep(0, 20)
new_beta <- c(0, beta)
names(new_beta) <- c("(Intercept)", paste0("X", 1:20))

for (i in seq_along(squared_coefficient_differences)) {
  coefs <- coef(best_subsets, id = i)
  missing_coefs <- setdiff(names(new_beta), names(coefs))

  if (length(missing_coefs) > 0) {
    missing_vector <- rep(0, length(missing_coefs)) %>% setNames(., missing_coefs)
    coefs <- c(coefs, missing_vector)
  }

  coefs <- coefs[names(new_beta)]
  squared_coefficient_differences[[i]] <- sqrt(sum(coefs - new_beta) ^ 2)
}

sum(beta > 0)
# 10
which.min(squared_coefficient_differences)
# 10


# Give the same results

coef(best_subsets, id = 10)
