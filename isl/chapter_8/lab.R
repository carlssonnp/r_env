library(ISLR)
library(tree)
library(randomForest)
library(gbm)
library(MASS)

Carseats$High <- as.factor(ifelse(Carseats$Sales <= 8, "No", "Yes"))

model <- tree(High ~ . - Sales, data = Carseats)

summary(model)

# Classification tree:
# tree(formula = High ~ . - Sales, data = Carseats)
# Variables actually used in tree construction:
# [1] "ShelveLoc"   "Price"       "Income"      "CompPrice"   "Population"
# [6] "Advertising" "Age"         "US"
# Number of terminal nodes:  27
# Residual mean deviance:  0.4575 = 170.7 / 373
# Misclassification error rate: 0.09 = 36 / 400


# Residual mean deviance is deviance divided by "degrees of freedom" of residuals


plot(model)
text(model, pretty = 0)


train_idx <- sample(1:nrow(Carseats), 200)

df_train <- Carseats[train_idx, ]
df_test <- Carseats[-train_idx, ]

model <- tree(High ~ . - Sales, data = df_train)

preds <- predict(model, df_test, "class")

table(preds, df_test$High, dnn = c("predicted", "actual"))

#            actual
# predicted  No Yes
# No  100  25
# Yes  21  54


cv_model <- cv.tree(model, FUN = prune.misclass)

cv_model

# $size
# [1] 22 18 14 11  7  4  3  2  1
#
# $dev
# [1] 66 67 71 71 70 58 55 55 85
#
# $k
# [1]      -Inf  0.000000  1.000000  1.333333  1.500000  2.666667  3.000000
# [8] 11.000000 31.000000
#
# $method
# [1] "misclass"
#
# attr(,"class")
# [1] "prune"         "tree.sequence"


# 2 is the optimal number of nodes


best_model <- prune.misclass(model, best = 9)

preds <- predict(best_model, df_test, "class")

table(preds, df_test$High, dnn = c("predicted", "actual"))

# predicted  No Yes
#       No  110  52
#       Yes  11  27



train_idx <- sample(1:nrow(Boston), nrow(Boston) %/% 2)

df_train <- Boston[train_idx, ]
df_test <- Boston[-train_idx, ]

model <- tree(medv ~ . , data = df_train)

cv_model <- cv.tree(model)


pruned_model <- prune.tree(model, best = 9)

preds <- predict(pruned_model, df_test)

sqrt(mean((preds - df_test$medv) ^ 2))

# [1] 5.715485

model <- randomForest(medv ~ ., data = df_train, mtry = 13, importance = TRUE)

preds <- predict(model, df_test)

sqrt(mean((preds - df_test$medv) ^ 2))
# [1] 4.04295


model <- randomForest(medv ~ ., data = df_train, importance = TRUE)

preds <- predict(model, df_test)

sqrt(mean((preds - df_test$medv) ^ 2))
# [1] 3.818545

# Better!

model <- gbm(medv ~ ., data = df_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

preds <- predict(model, df_test)

sqrt(mean((preds - df_test$medv) ^ 2))

# [1] 3.661172




p <- seq(0, 1, length = 100)

gini <- 2 * p * (1 - p)
entropy <- - p * log(p, 2)  - (1 - p) * log(1 - p, 2)
entropy[is.na(entropy)] <- 0
misclassification <- 1 - pmax(p, 1 - p)


df <- data.frame(
  p = rep(p, 3),
  value = c(gini, entropy, misclassification),
  label = c(rep("gini", 100), rep("entropy", 100), rep("misclassification", 100))
)


ggplot(df) +
  geom_point(aes(x = p, y = value, color = label))


ntrees <- c(1, 100, 1000)
n_vars <- seq(13)
results <- matrix(0, length(ntrees), length(n_vars))

ggplot_results <- matrix(0, length(ntrees) * length(n_vars), 3)

counter <- 1
for (i in seq_along(ntrees)) {
  for (j in seq_along(n_vars)) {
    n_tree <- ntrees[[i]]
    n_var <- n_vars[[j]]
    model <- randomForest(medv ~ ., data = df_train, ntree = ntree, mtry = n_var)
    preds <- predict(model, df_test)
    rmse <- sqrt(mean((df_test$medv - preds) ^ 2))

    results[i, j] <- rmse
    ggplot_results[counter, 1] <- n_tree
    ggplot_results[counter, 2] <- n_var
    ggplot_results[counter, 3] <- rmse
    counter <- counter + 1
  }
}


persp(ntrees, n_vars, results, theta = 60, phi = 15, expand = 0.5, col = "lightblue")


ggplot_results <- data.frame(ggplot_results) %>%
  setNames(., c("n_trees", "n_vars", "rmse"))

ggplot(ggplot_results) +
  geom_smooth(aes(x = n_vars, y = rmse, color = as.factor(n_trees)))

# Number of splitting variables is more important than number of trees

train_idx <- sample(nrow(Carseats), nrow(Carseats) %/% 2)
df_train <- Carseats[train_idx, ]
df_test <- Carseats[-train_idx, ]

model <- tree(Sales ~ ., data = df_train)

plot(model)
text(model)

# shelve location is again the most important feature

preds <- predict(model, df_test)

mean(sqrt((preds - df_test$Sales) ^ 2))
# [1] 1.831902


cv_model <- cv.tree(model)

# best tree is most complex one


model <- randomForest(Sales ~ ., data = df_train, mtry = 10, importance = TRUE)


preds <- predict(model, df_test)

mean(sqrt((preds - df_test$Sales) ^ 2))

# [1] 1.266838

importance(model)


model <- randomForest(Sales ~ ., data = df_train, importance = TRUE)

preds <- predict(model, df_test)

mean(sqrt((preds - df_test$Sales) ^ 2))

# [1] 1.356858


# A little worse

mods <- lapply(seq(10), function(idx) randomForest(Sales ~ ., data = df_train, mtry = idx))


preds <- lapply(mods, function(model) predict(model, df_test))

lapply(preds, function(pred) mean(sqrt((pred - df_test$Sales) ^ 2)))


# pretty stable after 6

train_idx <- sample(nrow(OJ), 800)

df_train <- OJ[train_idx, ]
df_test <- OJ[-train_idx, ]


model <- tree(Purchase ~ ., data = df_train)

# Only two variables were actually used to make the tree

preds <- predict(model, df_test, type="class")

table(preds, df_test$Purchase, dnn = c("predicted", "actual"))


# test error rate is

1 - (142 + 83) / (270)
# [1] 0.1666667

# which is actually lower than the train error rate

cv_model <- cv.tree(model, FUN = prune.misclass)


# Best model is unpruned

pruned <- prune.misclass(model, best = 5)


preds <- predict(pruned, df_test, type = "class")

table(preds, df_test$Purchase, dnn = c("predicted", "actual"))


# same test error, unpruned has lower train error


df_hitters <- Hitters
df_hitters <- df_hitters[!is.na(df_hitters$Salary), ]
df_hitters$Salary <- log(df_hitters$Salary)


df_train <- df_hitters[1:200, ]
df_test <- df_hitters[201:nrow(df_hitters), ]

etas <- c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

models <- lapply(etas, function(eta) gbm(Salary ~ ., data = df_train, n.trees = 1000, shrinkage = eta, distribution = "gaussian", ))


errors <- lapply(models, function(model) model$train.error[[1000]])

# train error always goes down as we increase eta

preds <- lapply(models, function(model) predict(model, df_test))

lapply(preds, function(pred) mean((pred - df_test$Salary) ^ 2))


# Starts to increase with larger eta


model <- randomForest(Salary ~ ., data = df_train, mtry = 19)

preds <- predict(model, df_test)

mean(sqrt((preds - df_test$Salary) ^ 2))

# [1] 0.3110401

# Worse than boosting


Caravan$Purchase <- ifelse(Caravan$Purchase == "Yes", 1, 0)
df_train <- Caravan[1:1000, ]
df_test <- Caravan[1001:nrow(Caravan), ]

model <- gbm(Purchase ~ ., data = df_train, n.trees = 1000, shrinkage = 0.01)

preds <- predict(model, df_test, type = "response")

classes <- ifelse(preds >= 0.2, 1, 0)

table(classes, df_test$Purchase, dnn = c("predicted", "actual"))

# did a good job

model <- randomForest(as.factor(Purchase) ~ ., data = df_train, ntree = 1000)

preds <- predict(model, type = "prob")
