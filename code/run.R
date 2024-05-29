### Data from: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success


library(randomForest)
library(adabag)
library(caret)
library(e1071)
library(UBL)
library(dplyr)
library(data.table)
source("Funktions.R")

real_data <- read.csv("data.csv",header = TRUE, sep = ";")
real_data <- as.data.table(real_data)

set.seed(12345)

C.perc <- "balance"
results_real_b <- replicate(100, test_function(real_data, "Target"))
C.perc <- "extreme"
results_real_e <- replicate(100, test_function(real_data, "Target"))



set.seed(12346)

data1 <- generate_imbalanced_multiclass_data()
#categories = c("aa","bb","cc","dd"), N = 1000, 
#class_ratios = c(0.05, 0.15, 0.5, 0.3), num_continuous = 3, 
#num_categorical = 2, noise_level = 0.1
data2 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc"),
                                             N = 1000, class_ratios = c(0.05, 0.45, 0.5), 
                                             num_continuous = 2, num_categorical = 2, 
                                             noise_level = 0.1)
data3 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd", "ee"),
                                                      N = 1000, class_ratios = c(0.25, 0.15, 0.05, 0.4, 0.15), 
                                                      num_continuous = 2, num_categorical = 4, 
                                                      noise_level = 0.1)
data4 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd", "ee"),
                                             N = 1000, class_ratios = c(0.2, 0.2, 0.2, 0.2, 0.2), 
                                             num_continuous = 2, num_categorical = 4, 
                                             noise_level = 0.1)
data5 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc"),
                                             N = 1000, class_ratios = c(0.2, 0.1, 0.7), 
                                             num_continuous = 2, num_categorical = 4, 
                                             noise_level = 0.1)
data6 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd"),
                                             N = 1000, class_ratios = c(0.2, 0.4, 0.1, 0.3), 
                                             num_continuous = 2, num_categorical = 4, 
                                             noise_level = 0.1)
data7 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd", "ee"),
                                             N = 1000, class_ratios = c(0.2, 0.2, 0.2, 0.2, 0.2), 
                                             num_continuous = 5, num_categorical = 1, 
                                             noise_level = 0.1)
data8 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc"),
                                             N = 1000, class_ratios = c(0.45, 0.15, 0.4), 
                                             num_continuous = 2, num_categorical = 2, 
                                             noise_level = 0.1)
data9 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd"),
                                             N = 1000, class_ratios = c(0.05,0.05,0.3,0.6), 
                                             num_continuous = 3, num_categorical = 3, 
                                             noise_level = 0.1)
data10 <- generate_imbalanced_multiclass_data(categories = c("aa","bb","cc","dd", "ee"),
                                             N = 1000, class_ratios = c(0.25,0.15,0.4,0.1,0.1), 
                                             num_continuous = 8, num_categorical = 4, 
                                             noise_level = 0.1)

data_ls <- list(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10)


C.perc <- "balance"
results_test_b <- sapply(data_ls, test_function, target_column = "targets")
results_test1_b <- replicate(100, test_function(data9, "targets"))


C.perc <- "extreme"
results_test_e <- sapply(data_ls, test_function, target_column = "targets")
results_test1_e <- replicate(100, test_function(data9, "targets"))



png("real_data_100_b.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_real_b),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
        col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_real_b), label = round(rowMeans(results_real_b),3), pos = 3, cex = 0.8)

dev.off()

png("test_data_10_b.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test_b),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test_b), label = round(rowMeans(results_test_b),2), pos = 3, cex = 0.8)

dev.off()

png("real_data_100_e.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_real_e),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_real_e), label = round(rowMeans(results_real_e),3), pos = 3, cex = 0.8)

dev.off()

png("test_data_10_e.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test_e),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test_e), label = round(rowMeans(results_test_e),2), pos = 3, cex = 0.8)

dev.off()

png("test_data_100_b.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test1_b),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test1_b), label = round(rowMeans(results_test1_b),2), pos = 3, cex = 0.8)

dev.off()

png("test_data_100_e.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test1_e),main = "Mean F1 Scores of different Models",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test1_e), label = round(rowMeans(results_test1_e),2), pos = 3, cex = 0.8)

dev.off()


save(results_real_b, file = "results_real_b.RData")
save(results_test1_b, file = "results_test1_b.RData")
save(results_test1_e, file = "results_test1_e.RData")
save(results_test_e, file = "results_test_e.RData")
save(results_real_e, file = "results_real_e.RData")
save(results_test_b, file = "results_test_b.RData")
