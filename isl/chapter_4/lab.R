library(dplyr)
library(ISLR)
library(MASS)
library(xgboost)
# Includes KNN
library(class)

df_markets <- Smarket

cor(df_markets %>% dplyr::select(., -Direction))

# Year         Lag1         Lag2         Lag3         Lag4
# Year   1.00000000  0.029699649  0.030596422  0.033194581  0.035688718
# Lag1   0.02969965  1.000000000 -0.026294328 -0.010803402 -0.002985911
# Lag2   0.03059642 -0.026294328  1.000000000 -0.025896670 -0.010853533
# Lag3   0.03319458 -0.010803402 -0.025896670  1.000000000 -0.024051036
# Lag4   0.03568872 -0.002985911 -0.010853533 -0.024051036  1.000000000
# Lag5   0.02978799 -0.005674606 -0.003557949 -0.018808338 -0.027083641
# Volume 0.53900647  0.040909908 -0.043383215 -0.041823686 -0.048414246
# Today  0.03009523 -0.026155045 -0.010250033 -0.002447647 -0.006899527
#   Lag5      Volume        Today
# Year    0.029787995  0.53900647  0.030095229
# Lag1   -0.005674606  0.04090991 -0.026155045
# Lag2   -0.003557949 -0.04338321 -0.010250033
# Lag3   -0.018808338 -0.04182369 -0.002447647
# Lag4   -0.027083641 -0.04841425 -0.006899527
# Lag5    1.000000000 -0.02200231 -0.034860083
# Volume -0.022002315  1.00000000  0.014591823
# Today  -0.034860083  0.01459182  1.000000000

# We see that the correlation between year and volume is positive.

logistic_regression_model <- glm(Direction ~ . -Today - Year, data = df_markets, family = "binomial")
lr_model_summary <- summary(logistic_regression_model)

1 - pchisq(lr_model_summary$null.deviance - lr_model_summary$deviance, df = nrow(coef(lr_model_summary)) - 1)
# [1] 0.7318693

# So it is likely that all the true coefficients are 0.

df_markets$pred_logistic <- predict(logistic_regression_model, type = "response")
# Default is on linear scale of predictors
df_markets$label_logistic <- ifelse(df_markets$pred_logistic >= 0.5, "Up", "Down")

table(df_markets$label_logistic, df_markets$Direction, dnn = c("Predicted", "Actual"))
#
#           Actual
# Predicted Down  Up
# Down  145 141
# Up    457 507

# Confusion matrix

total_accuracy <- (145 + 507) / 1250
# [1] 0.5216

# Or
mean(df_markets$label_logistic == df_markets$Direction)
# [1] 0.5216

# General R squared is 1 - D/D0, where D is the deviance between the fitted model and
# the saturated model, and D0 is the deviance between the model with only an intercept and the
# saturated model

df_markets <- Smarket

train_index <- df_markets$Year < 2005
train <- df_markets[train_index, ]
test <- df_markets[!train_index, ]

logistic_regression_model <- glm(Direction ~ . -Today - Year, data = train, family = "binomial")
test$pred_logistic <- predict(logistic_regression_model, test, type = "response")
test$label_logistic <- ifelse(test$pred_logistic >= 0.5, "Up", "Down")

table(test$label_logistic, test$Direction, dnn = c("Predicted", "Actual"))
#           Actual
# Predicted Down Up
# Down   77 97
# Up     34 44

mean(test$label_logistic == test$Direction)
# [1] 0.4801587

# Worse than random. High bias and non-null variance, so is to be expected

# Subset to only the important variables

logistic_regression_model <- glm(Direction ~ Lag1 + Lag2, data = train, family = "binomial")
test$pred_logistic <- predict(logistic_regression_model, test, type = "response")
test$label_logistic <- ifelse(test$pred_logistic >= 0.5, "Up", "Down")

mean(test$Direction== test$label_logistic)
# [1] 0.5595238

table(test$label_logistic, test$Direction, dnn = c("Predicted", "Actual"))

#           Actual
# Predicted Down  Up
#      Down   35  35
#      Up     76 106
# Better!


# True positive rate (recall, power, 1 - Type 11 error, sensitivity)

106 / (106 + 35)
# [1] 0.751773

# Positive predicted value (precision, 1 - false discovery proportion)
# (TP / (TP + FP))
106 / (106 + 76)
# [1] 0.5824176


# Now use LDA

lda_model <- MASS::lda(Direction ~ Lag1 + Lag2, data = train)
# Call:
# lda(Direction ~ Lag1 + Lag2, data = train)
#
# Prior probabilities of groups:
#     Down       Up
# 0.491984 0.508016
#
# Group means:
#             Lag1        Lag2
# Down  0.04279022  0.03389409
# Up   -0.03954635 -0.03132544
#
# Coefficients of linear discriminants:
#             LD1
# Lag1 -0.6420190
# Lag2 -0.5135293


preds <- predict(lda_model, test)

mean(preds$class == test$Direction)
# [1] 0.5595238

table(preds$class, test$Direction, dnn = c("Predicted", "Actual"))
#     Actual
# Predicted Down  Up
# Down   35  35
# Up     76 106

# Quadratic Discriminant Analysis

qda_model <- MASS::qda(Direction ~ Lag1 + Lag2, data = train)

preds <- predict(qda_model, test)

mean(preds$class == test$Direction)
# [1] 0.5992063

# Better!

table(preds$class, test$Direction, dnn = c("Predicted", "Actual"))

# Actual
# Predicted Down  Up
# Down   30  20
# Up     81 121


# Suggests a non-linear decision boundary
d_train <- xgboost::xgb.DMatrix(data = as.matrix(train[c("Lag1", "Lag2")]), info = list(label = ifelse(train$Direction == "Up", 1, 0)))
d_test <- xgboost::xgb.DMatrix(data = as.matrix(test[c("Lag1", "Lag2")]))

xgb_model <- xgboost::xgb.train(params = list(objective = "binary:logistic", max_depth="3", eta=0.3), data=d_train, nrounds=5)

preds <- predict(xgb_model, d_test)
labels <- ifelse(preds >= 0.5, "Up", "Down")

mean(labels == test$Direction)

# [1] 0.5793651

# Should really scale before KNN

train_matrix <- as.matrix(train[c("Lag1", "Lag2")])
test_matrix <- as.matrix(test[c("Lag1", "Lag2")])
train_label <- train$Direction

knn_preds <- class::knn(train_matrix, test_matrix, train_label, k=1)

mean(knn_preds == test$Direction)
# [1] 0.5

# Overfit
knn_preds <- class::knn(train_matrix, test_matrix, train_label, k=3)

mean(knn_preds == test$Direction)
# [1] 0.5277778

# Better


# Caravan data

df_caravan <- Caravan

# Gives us a matrix
standardized_caravan <- scale(Caravan %>% dplyr::select(., -Purchase))

train_idx <- 1:1000

train_x <- standardized_caravan[train_idx, ]
test_x <- standardized_caravan[-train_idx, ]
train_y <- df_caravan$Purchase[train_idx]
test_y <- df_caravan$Purchase[-train_idx]

preds <- class::knn(train_x, test_x, train_y, k = 1)

mean(preds == test_y)
# [1] 0.8890502

# 88 percent accuracy
# but note that just predicting no for everyone would result in accuracy of 94%

mean(test_y == "No")
# [1] 0.9400664


table(preds, test_y, dnn = c("Predicted", "Actual"))

# positive predicted value:

38 / (38 + 284)

# [1] 0.1180124


# Would be six if we predicted everyone to buy


preds <- class::knn(train_x, test_x, train_y, k = 3)

mean(preds == test_y)
# [1] 0.9253422

table(preds, test_y, dnn = c("Predicted", "Actual"))

#          Actual
# Predicted   No  Yes
# No  4437  264
# Yes   96   25


# positive predicted value of
25 / (25 + 96)

# 0.2066116


preds <- class::knn(train_x, test_x, train_y, k = 5)

mean(preds == test_y)
# [1] 0.9365409
table(preds, test_y, dnn = c("Predicted", "Actual"))

  # Actual
# Predicted   No  Yes
# No  4506  279
# Yes   27   10

# positive predicted value of
10 / (10 + 27)

# [1] 0.2702703

logistic_regression_model <- glm(Purchase ~ ., data = df_caravan, family = "binomial" )

# Can play around with the cutoff to get a better PPV
