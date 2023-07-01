cross_validate <- function(df, k) {

  indices <- sample(nrow(df))

  size <- nrow(df) %/% k

  df_list <- vector("list", k)
  index <- 1
  for (i in seq_along(df_list)) {
    df_list[[i]] <- df[indices[index:(index - 1 + size)], ]
    index <- index + size
  }
  # assign the remaining observations randomly
  remaining_obs <- nrow(df) - k * size

  if (remaining_obs > 0) {
    folds <- sample(length(df_list), remaining_obs)
    index <- k * size
    for (i in seq(remaining_obs)) {
      df_list[[folds[[i]]]] <- rbind(df_list[[folds[[i]]]], df[index + 1, ])
      index <- index + 1
    }
  }



  df_list

}

cross_validate_2 <- function(df, k) {
  indices <- rep(seq(k), length.out = nrow(df))
  indices <- sample(indices)

  split(df, indices)
}


cross_validate_3 <- function(df, k) {
  indices <- (seq(nrow(df)) %% k)
  indices <- sample(indices)

  split(df, indices)

}


simple_split <- function(df, k) {
  indices <- sort(rep(1:k, length.out = nrow(df)))

  split(df, indices)
}


# Lab

library(ISLR)
library(glmnet)
library(leaps)

df_hitters <- Hitters
df_hitters <- df_hitters[!is.na(df_hitters$Salary), ]


best_subset_models <- leaps::regsubsets(Salary ~ ., data = df_hitters, nvmax = 20)


summary_object <- summary(best_subset_models)

coef(best_subset_models, 1)

# (Intercept)        CRBI
# 274.5803864   0.7909536

coef(best_subset_models, 10)
# (Intercept)        AtBat         Hits        Walks       CAtBat        CRuns
# 162.5354420   -2.1686501    6.9180175    5.7732246   -0.1300798    1.4082490
#        CRBI       CWalks    DivisionW      PutOuts      Assists
#   0.7743122   -0.8308264 -112.3800575    0.2973726    0.2831680


# Doesn't return actual model object
df_results <- data.frame(n_vars = 1:19, bic = summary_object$bic)

best_model <- df_results[which.min(df_results$bic), ]

plot(best_subset_models)

ggplot(data = df_results) +
  geom_line(aes(x = n_vars, y = bic)) +
  geom_point(data = df_results[which.min(df_results$bic), ], aes(x = n_vars, y = bic), color="red")


# minimum occurs at 6 variable model


cross_validate <- function(df, k) {
  indices <- rep(seq(k), length.out = nrow(df))
  indices <- sample(indices)

  split(df, indices)
}


find_best_subset <- function(df, k) {
  df_list <- cross_validate(df, k)

  results <- matrix(0, nrow=k, ncol=19)
  for (i in seq_along(df_list)) {
    test <- df_list[[i]]
    train <- do.call(rbind, df_list[-i])

    best_subsets <- leaps::regsubsets(Salary ~ ., data = train, nvmax = 19)
    for (j in 1:19) {
      coefs <- coef(best_subsets, j)
      test_matrix <- model.matrix(Salary ~ ., data = test)
      prediction <- test_matrix[, names(coefs)] %*% coefs

      results[i, j] <- mean((prediction - test$Salary) ^ 2)
    }

  }

  apply(results, 2, mean)
}

results <- find_best_subset(df_hitters, 10)


which.min(results)

# [1] 11

min(results)
# [1] 114795.9

# so the 11 variable model is best


final_model <- leaps::regsubsets(Salary ~ ., data = df_hitters, nvmax = 19)


coef(final_model, 11)

# (Intercept)        AtBat         Hits        Walks       CAtBat        CRuns
# 135.7512195   -2.1277482    6.9236994    5.6202755   -0.1389914    1.4553310
#        CRBI       CWalks      LeagueN    DivisionW      PutOuts      Assists
#   0.7852528   -0.8228559   43.1116152 -111.1460252    0.2894087    0.2688277





# Ridge regression
# alpha is the weight applied to LASSO

# Lasso = least absolute shrinkage and selection operator

x <- model.matrix(Salary ~ ., df_hitters)[, -1]
y <- df_hitters$Salary

grid <- 10 ^ seq(10, -2, length = 100)

ridge_model <- glmnet::glmnet(x, y, alpha = 0, lambda = grid)


dim(coef(ridge_model))
# [1]  20 100


predict(ridge_model, s=50, type = "coefficients")

# 20 x 1 sparse Matrix of class "dgCMatrix"
#                        s1
# (Intercept)  4.876610e+01
# AtBat       -3.580999e-01
# Hits         1.969359e+00
# HmRun       -1.278248e+00
# Runs         1.145892e+00
# RBI          8.038292e-01
# Walks        2.716186e+00
# Years       -6.218319e+00
# CAtBat       5.447837e-03
# CHits        1.064895e-01
# CHmRun       6.244860e-01
# CRuns        2.214985e-01
# CRBI         2.186914e-01
# CWalks      -1.500245e-01
# LeagueN      4.592589e+01
# DivisionW   -1.182011e+02
# PutOuts      2.502322e-01
# Assists      1.215665e-01
# Errors      -3.278600e+00
# NewLeagueN  -9.496680e+00


# Does interpolation

nrows <- nrow(x)
train_idx <- sample(nrows, nrows %/% 2)
# or
# train_idx <- sample(c(TRUE, FALSE), replace = TRUE)

x_train <- x[train_idx, ]
x_test <- x[-train_idx, ]

y_train <- df_hitters[train_idx, "Salary"]
y_test <- df_hitters[-train_idx, "Salary"]


ridge_model <- glmnet::glmnet(x_train, y_train, lambda = grid, alpha = 0)


preds <- predict(ridge_model, s = 4, newx = x_test)
mean((preds - y_test) ^ 2)
# [1] 124965.6

# vs

mean((mean(y_test) - y_test) ^ 2)
# [1] 210493.5


# vs

df_train <- data.frame(x_train)
df_train$Salary <- y_train

df_test <- data.frame(x_test)
df_test$Salary <- y_test

ols_model <- lm(Salary ~ ., data = df_train)

preds <- predict(ols_model, newdata = df_test)

mean((preds - y_test) ^ 2)
# [1] 136385.2

# Larger!

# should be same as
preds <- predict(ridge_model, s = 0, newx = x_test, exact = TRUE, x = x_train, y = y_train)
mean((preds - y_test) ^ 2)
# [1] 135623.6


# Or get it the cross-validated way

set.seed(1)
cv_model <- glmnet::cv.glmnet(x_train, y_train, alpha = 0, lambda = grid)

cv_model$lambda.min
# [1] 466.566

cv_model$lambda.1se
# [1] 2999.12

preds <- predict(ridge_model, s = cv_model$lambda.min, newx = x_test)
mean((preds - y_test) ^ 2)



# Lasso

lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = grid)
plot(lasso_model)

cv_model <- cv.glmnet(x_train, y_train, alpha = 1)


lambda_min <- cv_model$lambda.min

lambda_min
# [1] 23.2211

preds <- predict(lasso_model, s = lambda_min, newx = x_test)

mean((preds - y_test) ^ 2)
# [1] 133099.7


# Not as good as ridge, better than OLS

full <- glmnet(x, y, alpha = 1, lambda = grid)

predict(full, s = lambda_min, type = "coefficients")

# s1
# (Intercept)  33.7854295
# AtBat         .
# Hits          1.8122361
# HmRun         .
# Runs          .
# RBI           .
# Walks         2.1418985
# Years         .
# CAtBat        .
# CHits         .
# CHmRun        .
# CRuns         0.2016389
# CRBI          0.4033111
# CWalks        .
# LeagueN       .
# DivisionW   -91.8434190
# PutOuts       0.2042033
# Assists       .
# Errors        .
# NewLeagueN    .

# We see that the lasso performs variable selection


# PCR and PLS
library(pls)
library(ISLR)

df_hitters <- Hitters
df_hitters <- df_hitters[!is.na(df_hitters$Salary), ]

nrows <- nrow(df_hitters)
train_idx <- sample(nrows, nrows %/% 2)

df_train <- df_hitters[train_idx, ]
df_test <- df_hitters[-train_idx, ]

pcr_model <- pcr(Salary ~ ., data = df_hitters, validation = "CV", scale = TRUE)


validationplot(pcr_model, val.type = "MSEP")

# 1 component performs about as well as all of them

pcr_model <- pcr(Salary ~ ., data = df_train, validation = "CV", scale = TRUE)

validationplot(pcr_model, val.type="MSEP")


predictions <- predict(pcr_model, df_test, ncomp = 1)

mean((df_test$Salary - predictions) ^ 2)
# [1] 102298.1

# Not bad !

# PLS
pls_model <- plsr(Salary ~ ., data = df_train, validation = "CV", scale = TRUE)

validationplot(pls_model, val.type = "MSEP")

# 1 performs very well

predictions <- predict(pls_model, df_test, ncomp = 1)

mean((df_test$Salary - predictions) ^ 2)
# [1] 98394.17



# Note that that the variance explained of X is lower than PCA, but the variance explained of
# Y is higher. Higher variance, lower bias. Think of PLS as supervised dimension reduction. 
