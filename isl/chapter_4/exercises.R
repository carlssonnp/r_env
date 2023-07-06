library(ISLR)
library(GGally)
library(MASS)
library(class)

df_weekly <- Weekly

pairwise_plot <- GGally::ggpairs(df_weekly, columns=c("Volume", paste0("Lag", 1:2), "Direction"))

# Not much correlation

logistic_regression_model <- glm(
  Direction ~ Volume + Lag1 + Lag2 + Lag3 + Lag4 + Lag5,
  data = df_weekly,
  family = "binomial"
)

print(summary(logistic_regression_model))


# Call:
# glm(formula = Direction ~ Volume + Lag1 + Lag2 + Lag3 + Lag4 +
#     Lag5, family = "binomial", data = df_weekly)
#
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)
# (Intercept)  0.26686    0.08593   3.106   0.0019 **
# Volume      -0.02274    0.03690  -0.616   0.5377
# Lag1        -0.04127    0.02641  -1.563   0.1181
# Lag2         0.05844    0.02686   2.175   0.0296 *
# Lag3        -0.01606    0.02666  -0.602   0.5469
# Lag4        -0.02779    0.02646  -1.050   0.2937
# Lag5        -0.01447    0.02638  -0.549   0.5833
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# (Dispersion parameter for binomial family taken to be 1)
#
#     Null deviance: 1496.2  on 1088  degrees of freedom
# Residual deviance: 1486.4  on 1082  degrees of freedom
# AIC: 1500.4
#
# Number of Fisher Scoring iterations: 4

# Lag two is significant
preds <- predict(logistic_regression_model, type = "response")
labels <- ifelse(preds > 0.5, "Up", "Down")

table(labels, df_weekly$Direction, dnn = c("Predicted", "Actual"))


#          Actual
# Predicted Down  Up
# Down   54  48
# Up    430 557


mean(labels == df_weekly$Direction)
# [1] 0.5610652

# The True positive rate is good (sensitivity), but the true negative rate is bad (specificity)
# Positive predicted value (precision) is over 50%
# negative predicted value is over 50%


train_index <- df_weekly$Year <= 2008

train <- df_weekly[train_index, ]
test <- df_weekly[!train_index, ]

logistic_regression_model <- glm(Direction ~ Lag2, data = train, family = "binomial")

preds <- predict(logistic_regression_model, test, type = "response")
labels <- ifelse(preds >= 0.5, "Up", "Down")

mean(test$Direction == labels)
# [1] 0.625

table(labels, test$Direction, dnn = c("Predicted", "Actual"))

#           Actual
# Predicted Down Up
# Down    9  5
# Up     34 56

# Same problem; good true positive rate = recall = power = sensitivity = 1 - type 2 error,
# bad true negative rate


lda_model <- lda(Direction ~ Lag2, data = train)

preds <- predict(lda_model, test)

mean(test$Direction == preds$class)
# [1] 0.625

table(preds$class, test$Direction, dnn = c("Predicted", "Actual"))

#       Actual
# Predicted Down Up
# Down    9  5
# Up     34 56


# Same as logistic

qda_model <- qda(Direction ~ Lag2, data = train)

preds <- predict(qda_model, test)

mean(test$Direction == preds$class)
# [1] 0.5865385

table(preds$class, test$Direction, dnn = c("Predicted", "Actual"))

# Actual
# Predicted Down Up
# Down    0  0
# Up     43 61

# Predicts up all the time


train_matrix <- as.matrix(train[, "Lag2"])
test_matrix <- as.matrix(test[, "Lag2"])


predicted_labels <- class::knn(as.matrix(train_matrix), test_matrix, train$Direction, 1)


table(predicted_labels, test$Direction, dnn = c("Predicted", "Actual"))

# Actual
# Predicted Down Up
# Down   21 29
# Up     22 32

mean(predicted_labels == test$Direction)

# [1] 0.5096154

# Terrible

# logistic and lda seem to have similar error rates
