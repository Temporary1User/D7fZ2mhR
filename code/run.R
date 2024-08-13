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


save(results_real_b, file = "results_real_b.RData")
save(results_test1_b, file = "results_test1_b.RData")
save(results_test1_e, file = "results_test1_e.RData")
save(results_test_e, file = "results_test_e.RData")
save(results_real_e, file = "results_real_e.RData")
save(results_test_b, file = "results_test_b.RData")



results_real_e <- as.data.frame(results_real_e)
results_real_b <- as.data.frame(results_real_b)
results_test1_b <- as.data.frame(results_test1_b)
results_test1_e <- as.data.frame(results_test1_e)
results_test_e <- as.data.frame(results_test_e)
results_test_b <- as.data.frame(results_test_b)


row.names(results_real_e) <- c("RF", "Balanced_RF", "eSMOTE_RF", "SVM", "eSMOTE_SVM", "Ada", "eSMOTE_Ada")
row.names(results_test1_e) <- c("RF", "Balanced_RF", "eSMOTE_RF", "SVM", "eSMOTE_SVM", "Ada", "eSMOTE_Ada")
row.names(results_test_e) <- c("RF", "Balanced_RF", "eSMOTE_RF", "SVM", "eSMOTE_SVM", "Ada", "eSMOTE_Ada")
results_real <- rbind(results_real_b, results_real_e[c("eSMOTE_RF", "eSMOTE_SVM", "eSMOTE_Ada"),])
results_test1 <- rbind(results_test1_b, results_test1_e[c("eSMOTE_RF", "eSMOTE_SVM", "eSMOTE_Ada"),])
results_test <- rbind(results_test_b, results_test_e[c("eSMOTE_RF", "eSMOTE_SVM", "eSMOTE_Ada"),])

data_with_balenceTech <- c("Balanced_RF", "SMOTE_RF","eSMOTE_RF","SMOTE_SVM","eSMOTE_SVM", "SMOTE_Ada","eSMOTE_Ada")
data_without_balenceTech <- c("RF", "SVM", "Ada")

png("real_data_without_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_real[data_without_balenceTech,]),main = "Mean F1 Scores of different Models with real world data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_real[data_without_balenceTech,]), label = round(rowMeans(results_real[data_without_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

png("real_data_with_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_real[data_with_balenceTech,]),main = "Mean F1 Scores of different Models with real world data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_real[data_with_balenceTech,]), label = round(rowMeans(results_real[data_with_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

real_data <- rowMeans(results_real)
real_data <- real_data[order(real_data)]

png("real_data.png", width = 2500, height = 1500, res = 300)

bp <- barplot(real_data, main = "Mean F1 Scores of different Models with real world data", xlab = "Model", ylab = "F1 Score",
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = real_data, label = round(real_data, 3), pos = 3, cex = 0.8)

dev.off()


png("test_data_100_without_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test1[data_without_balenceTech,]),main = "Mean F1 Scores of different Models with generated data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test1[data_without_balenceTech,]), label = round(rowMeans(results_test1[data_without_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

png("test_data_100_with_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test1[data_with_balenceTech,]),main = "Mean F1 Scores of different Models with generated data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test1[data_with_balenceTech,]), label = round(rowMeans(results_test1[data_with_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

test_data_100 <- rowMeans(results_test1)
test_data_100 <- test_data_100[order(test_data_100)]

png("test_data_100.png", width = 2500, height = 1500, res = 300)

bp <- barplot(test_data_100, main = "Mean F1 Scores of different Models with generated data", xlab = "Model", ylab = "F1 Score",
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = test_data_100, label = round(test_data_100, 3), pos = 3, cex = 0.8)

dev.off()



png("test_data_10_without_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test[data_without_balenceTech,]),main = "Mean F1 Scores of different Models with different generated data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test[data_without_balenceTech,]), label = round(rowMeans(results_test[data_without_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

png("test_data_10_with_BT.png", width = 2000, height = 1500, res = 300)

bp <- barplot(rowMeans(results_test[data_with_balenceTech,]),main = "Mean F1 Scores of different Models with different generated data",xlab = "Model", ylab = "F1 Score", 
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = rowMeans(results_test[data_with_balenceTech,]), label = round(rowMeans(results_test[data_with_balenceTech,]),3), pos = 3, cex = 0.8)

dev.off()

test_data_10 <- rowMeans(results_test)
test_data_10 <- test_data_10[order(test_data_10)]

png("test_data_10.png", width = 2500, height = 1500, res = 300)

bp <- barplot(test_data_10, main = "Mean F1 Scores of different Models with different generated data", xlab = "Model", ylab = "F1 Score",
              col = "gray", border = "black", width = 0.1, space = 0, ylim = c(0,1), cex.names = 0.5)
text(x = bp, y = test_data_10, label = round(test_data_10, 3), pos = 3, cex = 0.8)

dev.off()
