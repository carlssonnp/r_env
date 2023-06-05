library(magrittr)
library(dplyr)
library(GGally)
library(ggplot2)
library(tidyr)
library(MASS)


image_path <- "isl/chapter_2/images/"
# Write dataset to CSV
# library(ISLR)
#
# data("College")
#
# write.csv(College, "isl/chapter_2/data/college.csv")
#
# data("Auto")
#
# write.csv(Auto, "isl/chapter_2/data/auto.csv")


df_college <- read.csv("isl/chapter_2/data/college.csv")

summary(df_college)

# X               Private               Apps           Accept
# Length:777         Length:777         Min.   :   81   Min.   :   72
# Class :character   Class :character   1st Qu.:  776   1st Qu.:  604
# Mode  :character   Mode  :character   Median : 1558   Median : 1110
#                                  Mean   : 3002   Mean   : 2019
#                                  3rd Qu.: 3624   3rd Qu.: 2424
#                                  Max.   :48094   Max.   :26330
# Enroll       Top10perc       Top25perc      F.Undergrad
# Min.   :  35   Min.   : 1.00   Min.   :  9.0   Min.   :  139
# 1st Qu.: 242   1st Qu.:15.00   1st Qu.: 41.0   1st Qu.:  992
# Median : 434   Median :23.00   Median : 54.0   Median : 1707
# Mean   : 780   Mean   :27.56   Mean   : 55.8   Mean   : 3700
# 3rd Qu.: 902   3rd Qu.:35.00   3rd Qu.: 69.0   3rd Qu.: 4005
# Max.   :6392   Max.   :96.00   Max.   :100.0   Max.   :31643
# P.Undergrad         Outstate       Room.Board       Books
# Min.   :    1.0   Min.   : 2340   Min.   :1780   Min.   :  96.0
# 1st Qu.:   95.0   1st Qu.: 7320   1st Qu.:3597   1st Qu.: 470.0
# Median :  353.0   Median : 9990   Median :4200   Median : 500.0
# Mean   :  855.3   Mean   :10441   Mean   :4358   Mean   : 549.4
# 3rd Qu.:  967.0   3rd Qu.:12925   3rd Qu.:5050   3rd Qu.: 600.0
# Max.   :21836.0   Max.   :21700   Max.   :8124   Max.   :2340.0
# Personal         PhD            Terminal       S.F.Ratio
# Min.   : 250   Min.   :  8.00   Min.   : 24.0   Min.   : 2.50
# 1st Qu.: 850   1st Qu.: 62.00   1st Qu.: 71.0   1st Qu.:11.50
# Median :1200   Median : 75.00   Median : 82.0   Median :13.60
# Mean   :1341   Mean   : 72.66   Mean   : 79.7   Mean   :14.09
# 3rd Qu.:1700   3rd Qu.: 85.00   3rd Qu.: 92.0   3rd Qu.:16.50
# Max.   :6800   Max.   :103.00   Max.   :100.0   Max.   :39.80
# perc.alumni        Expend        Grad.Rate
# Min.   : 0.00   Min.   : 3186   Min.   : 10.00
# 1st Qu.:13.00   1st Qu.: 6751   1st Qu.: 53.00
# Median :21.00   Median : 8377   Median : 65.00
# Mean   :22.74   Mean   : 9660   Mean   : 65.46
# 3rd Qu.:31.00   3rd Qu.:10830   3rd Qu.: 78.00
# Max.   :64.00   Max.   :56233   Max.   :118.00

pairwise_plot <- GGally::ggpairs(data = df_college, columns = 2:5)

ggplot2::ggsave("isl/chapter_2/images/college_pairwise_plot.png", pairwise_plot)

box_plot_private <- ggplot2::ggplot(data = df_college) +
  ggplot2::geom_boxplot(ggplot2::aes(x = Private, y = Outstate, fill = Private))

ggplot2::ggsave("isl/chapter_2/images/college_boxplot_private.png", box_plot_private)

df_college <- df_college %>%
  dplyr::mutate(., elite = factor(ifelse(Top10perc > 50, "Yes", "No")))

table(df_college$elite)

# No Yes
# 699  78

box_plot_elite <- ggplot2::ggplot(data = df) +
  ggplot2::geom_boxplot(ggplot2::aes(x = elite, y = Outstate, fill = elite))

ggplot2::ggsave(paste0(image_path, "/college_boxplot_elite.png"), box_plot_elite)

df_auto <- read.csv("isl/chapter_2/data/auto.csv")

summary(df_auto)

# X               mpg          cylinders      displacement
# Min.   :  1.00   Min.   : 9.00   Min.   :3.000   Min.   : 68.0
# 1st Qu.: 99.75   1st Qu.:17.00   1st Qu.:4.000   1st Qu.:105.0
# Median :198.50   Median :22.75   Median :4.000   Median :151.0
# Mean   :198.52   Mean   :23.45   Mean   :5.472   Mean   :194.4
# 3rd Qu.:296.25   3rd Qu.:29.00   3rd Qu.:8.000   3rd Qu.:275.8
# Max.   :397.00   Max.   :46.60   Max.   :8.000   Max.   :455.0
# horsepower        weight      acceleration        year           origin
# Min.   : 46.0   Min.   :1613   Min.   : 8.00   Min.   :70.00   Min.   :1.000
# 1st Qu.: 75.0   1st Qu.:2225   1st Qu.:13.78   1st Qu.:73.00   1st Qu.:1.000
# Median : 93.5   Median :2804   Median :15.50   Median :76.00   Median :1.000
# Mean   :104.5   Mean   :2978   Mean   :15.54   Mean   :75.98   Mean   :1.577
# 3rd Qu.:126.0   3rd Qu.:3615   3rd Qu.:17.02   3rd Qu.:79.00   3rd Qu.:2.000
# Max.   :230.0   Max.   :5140   Max.   :24.80   Max.   :82.00   Max.   :3.000
# name
# Length:392
# Class :character
# Mode  :character


df_auto %>%
  dplyr::select_if(., is.numeric) %>%
  lapply(., range)
# $X
# [1]   1 397
#
# $mpg
# [1]  9.0 46.6
#
# $cylinders
# [1] 3 8
#
# $displacement
# [1]  68 455
#
# $horsepower
# [1]  46 230
#
# $weight
# [1] 1613 5140
#
# $acceleration
# [1]  8.0 24.8
#
# $year
# [1] 70 82
#
# $origin
# [1] 1 3

df_auto %>%
  dplyr::summarise_if(., is.numeric, list(sd = sd, mean = mean))

#   X_sd   mpg_sd cylinders_sd displacement_sd horsepower_sd weight_sd
# 1 114.4381 7.805007     1.705783         104.644      38.49116  849.4026
# acceleration_sd  year_sd origin_sd   X_mean mpg_mean cylinders_mean
# 1        2.758864 3.683737 0.8055182 198.5204 23.44592       5.471939
# displacement_mean horsepower_mean weight_mean acceleration_mean year_mean
# 1           194.412        104.4694    2977.584          15.54133  75.97959
# origin_mean
# 1    1.576531

df_auto %>%
  dplyr::filter(., !dplyr::between(dplyr::row_number(), 10, 85)) %>%
  dplyr::summarise_if(., is.numeric, list(sd = sd, mean = mean))


#   X_sd   mpg_sd cylinders_sd displacement_sd horsepower_sd weight_sd
# 1 96.81174 7.867283     1.654179        99.67837      35.70885  811.3002
# acceleration_sd  year_sd origin_sd   X_mean mpg_mean cylinders_mean
# 1        2.693721 3.106217   0.81991 234.6741 24.40443       5.373418
# displacement_mean horsepower_mean weight_mean acceleration_mean year_mean
# 1          187.2405        100.7215    2935.972           15.7269  77.14557
# origin_mean
# 1    1.601266


pairwise_plot_auto <- GGally::ggpairs(data=df_auto, columns=1:5)

ggplot2::ggsave(paste0(image_path, "/auto_pairwise_plot.png"), pairwise_plot_auto)


# Looks like displacement and mpg are negatively correlated, with maybe an inverse
# relationship

df_boston <- Boston

dim(df_boston)
# [1] 506  14

boston_pairs_plot <- GGally::ggpairs(df_boston, columns=1:10)

ggplot2::ggsave(paste0(image_path, "/boston_pairwise_plot.png"), boston_pairs_plot)

# High crime has low dis

# filter to top 10% of crimes
df_boston_top_10 <- df_boston %>%
  dplyr::filter(., dplyr::ntile(crim, 10) == 10)

# How many suburbs are bound by the Charles river?
sum(df_boston$chas)
# [1] 35

median(df_boston$ptratio)
# [1] 19.05

df_boston %>%
  dplyr::filter(., medv == min(medv))
#   crim zn indus chas   nox    rm age    dis rad tax ptratio  black lstat
# 1 38.3518  0  18.1    0 0.693 5.453 100 1.4896  24 666    20.2 396.90 30.59
# 2 67.9208  0  18.1    0 0.693 5.683 100 1.4254  24 666    20.2 384.97 22.98
# medv
# 1    5

# They have really high crime rates

sum(df_boston$rm > 7)
# [1] 64
sum(df_boston$rm > 8)
# [1] 13
