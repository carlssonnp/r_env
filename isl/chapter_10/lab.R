
pr <- prcomp(USArrests, scale = TRUE)
pr
# [1] 1.5748783 0.9948694 0.5971291 0.4164494
#
# Rotation (n x k) = (4 x 4):
#                 PC1        PC2        PC3         PC4
# Murder   -0.5358995 -0.4181809  0.3412327  0.64922780
# Assault  -0.5831836 -0.1879856  0.2681484 -0.74340748
# UrbanPop -0.2781909  0.8728062  0.3780158  0.13387773
# Rape     -0.5434321  0.1673186 -0.8177779  0.08902432

USArrests[] <- scale(USArrests)

svd(df_scaled)

biplot(pr, scale = 0)


# kmeans clustering

x <- matrix(rnorm(50 * 2), 50, 2)
x[1:25, 1] <- x[1:25, 1] + 3
x[1:25, 2] <- x[1:25, 2] - 4


model <- kmeans(x, 2, nstart = 20)

model <- hclust(dist(x), method = "complete")

plot(model)

cutree(model, 2)

#   [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  [38] 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1
#  [75] 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
# [112] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# [149] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# [186] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2


x <- matrix(rnorm(30 * 3), 30, 3)

dd <- as.dist(1 - t(cor(x)))


model <- hclust(dd, method = "complete")

nci_data <- NCI60$data
nci_labs <- NCI60$labs


model <- prcomp(nci_data, scale = TRUE)

colors <- function(vec) {
  cols <- rainbow(length(unique(vec)))
  cols[as.numeric(as.factor(vec))]
}


color_labels <- colors(nci_labs)

plot(model$x[, 1:2], col = color_labels)


# Generally observations with shared cancer subtypes have similar PC values

plot(model$x[, c(1, 3)], col = color_labels)
plot(model)


props <- model$sdev ^ 2 / sum(model$sdev ^ 2)


plot(props)

plot(cumsum(props))

plot(hclust(dist(nci_data)), labels = nci_labs)
plot(hclust(dist(nci_data), method = "average"), labels = nci_labs)
plot(hclust(dist(nci_data), method = "single"), labels = nci_labs)


# complete and average generally preferred

model <- hclust(dist(nci_data))

clusters <- cutree(model, 4)

table(clusters, nci_labs)


plot(model, labels = nci_labs)
abline(h = 139, col = "red")


model <- kmeans(nci_data, 4, nstart = 20)
clusters_k <- model$cluster


table(clusters, clusters_k)

# can perform both types of clustering on just some of the principal components


x <- matrix(c(1, 1, 0, 5, 6, 4, 4, 3, 4, 1, 2, 0), 6, 2)

plot(x)


labels <- sample(1:2, nrow(x), replace = TRUE)

changed <- TRUE
while (changed == TRUE) {
  old_labels <- labels
  print(1)
  centroids <- lapply(
    1:2,
    function(label) {
      obs <- x[labels == label, , drop = FALSE]
      apply(obs, 2, mean)
    }
  )

  for (i in seq_along(labels)) {
    distances <- lapply(
      centroids,
      function(centroid) {
        sum((x[i, ] - centroid) ^ 2)
      }
    )
    label <- which.min(distances)
    labels[[i]] <- label
  }

  changed <- !(identical(old_labels, labels))
}


mu_a <- 0
mu_b <- 0.5

sigma_a <- 1
sigma_b <- 2

machines <- list(
  a = list(mu = mu_a, sigma = sigma_a),
  b = list(mu = mu_b, sigma = sigma_b)
)

x <- matrix(0, 100, 1000)

probs <- seq(0, 1, length = 100)

machine_list <- c()

for (i in seq_along(probs)) {
  machine <- sample(2, 1, prob = c(probs[[i]], 1 - probs[[i]]))
  machine_list <- c(machine_list, machine)
  mu <- machines[[machine]]$mu
  sigma <- machines[[machine]]$sigma

  x[i, ] <- rnorm(1000, mu, sigma)
}


x_1 <- x[machine_list == 1, ]
x_2 <- x[machine_list == 2, ]

x <- rbind(scale(x_1), scale(x_2))

USArrests_scaled <- USArrests
USArrests_scaled[] <- apply(USArrests, 1, scale) %>% t

a <- 1 - cor(t(USArrests_scaled))
b <- as.matrix(dist(USArrests_scaled)) ^ 2


# differ by a ratio of 6

USArrests_scaled <- scale(USArrests)
model <- prcomp(USArrests_scaled)

cumsum(model$sdev ^ 2) / sum(model$sdev ^ 2)
# [1] 0.6200604 0.8675017 0.9566425 1.0000000



cumsum(diag(var(model$x))) / 4
# PC1       PC2       PC3       PC4
# 0.6200604 0.8675017 0.9566425 1.0000000


# Or
cumsum(diag(var(USArrests_scaled %*% model$rotation))) / 4

# PC1       PC2       PC3       PC4
# 0.6200604 0.8675017 0.9566425 1.0000000




model <- hclust(dist(USArrests_scaled), method = "complete")
plot(model)

cuts <- cutree(model, 3)

# Southern states are all together


model <- hclust(dist(USArrests), method = "complete")

cuts <- cutree(model, 3)

# Big difference; data should be scaled because they are not originally on the sane scale


x1 <- matrix(rnorm(20 * 50), 20, 50)
x2 <- matrix(rnorm(20 * 50, 10), 20, 50)
x3 <- matrix(rnorm(20 * 50, 20), 20, 50)

x <- rbind(x1, x2, x3)

labels <- c(rep(1, 20), rep(2, 20), rep(3, 20))


model <- prcomp(x)

pr_matrix <- model$x

plot(pr_matrix[, 1:2], col = labels)

model <- kmeans(x, 3, nstart = 20)


table(model$cluster, labels)

# perfect classfication


model <- kmeans(x, 2, nstart = 20)

# Collapsed two of the groups together

model <- kmeans(x, 4, nstart = 20)

# split the groups


model <- kmeans(model$x[, 1:2], 3, nstart = 20)

# perfect again


df <- read.csv("~/Downloads/Ch10Ex11.csv", header = FALSE)
df <- t(df)
labels <- c(rep(1, 20), rep(2, 20))

cor_dist <- as.dist(1 - cor(t(df)))

model <- hclust(cor_dist)

clusters <- cutree(model, 2)

table(clusters, labels)


df_1 <- df[clusters == 1, ]
df_2 <- df[clusters == 2, ]


tests <- rep(0, 1000)
for (i in seq_along(tests)) {
  tests[[i]] <- t.test(df_1[, i], df_2[, i])$p.value
}

plot(tests)

which.min(tests)


# grouping together some of observations incorrectly

model <- hclust(cor_dist, method = "average")


clusters <- cutree(model, 2)
