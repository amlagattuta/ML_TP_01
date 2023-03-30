

rm(list=ls())

set.seed(1693)
library(ISLR2)
library(e1071)
data(OJ)
OJ <- na.omit(OJ)


#### A ####
n <- nrow(OJ)
train_index <- sample(1:n, 800, replace = FALSE)

train_data <- OJ[train_index,]
test_data <- OJ[-train_index,]


#### B ####
svm_model <- svm(Purchase ~ ., data = train_data, cost = 0.01, kernel = "linear", type = "C-classification")

summary(svm_model)
####describe
     #432 support vectors
     #lots of support vectors because cost is lower, so margin is wider
     #evenly split

#### C ####
svm_train_preds <- predict(svm_model, train_data)
svm_train_error <- mean(svm_train_preds != train_data$Purchase)
svm_train_error
### train error: 0.17125

svm_test_preds <- predict(svm_model, test_data)
svm_test_error <- mean(svm_test_preds != test_data$Purchase)

svm_test_error
### test error: 0.1925926

cat("Training error rate:", svm_train_error, "\n")
cat("Test error rate:", svm_test_error, "\n")



#### D ####
tune.out <- tune(svm, Purchase ~ ., data = train_data, kernel = "linear", ranges = list(cost = c(0.01, 0.1, 1, 10)))
summary(tune.out)


#A higher value of C will lead to a smaller margin, which means the model is more likely to classify all training examples correctly but may overfit the data.
#A smaller value of C will result in a larger margin, which means the model may allow some training examples to be misclassified but will generalize better to test data.


best_cost <- tune.out$best.parameters$cost
best_cost
#### E ####

svm.fit2 <- svm(Purchase ~ ., data = train_data, kernel = "linear", cost = best_cost)
train.pred2 <- predict(svm.fit2, train_data)
test.pred2 <- predict(svm.fit2, test_data)

train_error_2 <- mean(train_data$Purchase != train.pred2)
test_error_2 <- mean(test_data$Purchase != test.pred2)

print(train_error_2)
#  0.16375
print(test_error_2)
#  0.1814815
### ever so slightly better, barely


#### F ####
svm_radial <- svm(Purchase ~ ., data = train_data, kernel = "radial", cost = 0.01)
summary(svm_radial)
### more vectors 

train.pred3 <- predict(svm_radial, train_data)
test.pred3 <- predict(svm_radial, test_data)

radial_train_error <- mean(train_data$Purchase != train.pred3)
radial_test_error <- mean(test_data$Purchase != test.pred3)

print(radial_train_error)
#0.3725
print(radial_test_error)
#0.4407407

#### not great, have to tune and do best cost again 
tune.out.radial <- tune(svm, Purchase ~ ., data = train_data, kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 10)))
summary(tune.out.radial)

best_cost_radial <- tune.out.radial$best.parameters$cost
best_cost_radial

svm_radial_best <- svm(Purchase ~ ., data = train_data, kernel = "radial", cost = best_cost_radial)
train_best_radial_preds <- predict(svm_radial_best, train_data)
test_best_radial_preds <- predict(svm_radial_best, test_data)

train_error_best_radial <- mean(train_data$Purchase != train_best_radial_preds)
test_error_best_radial <- mean(test_data$Purchase != test_best_radial_preds)

print(train_error_best_radial)
# 0.145
print(test_error_best_radial)
# 0.1925926

#### G #### 
svm_poly_fit <- svm(Purchase ~ ., data = train_data, kernel = "polynomial", degree = 2, cost = .01)

summary(svm_poly_fit)

pred_train_poly <- predict(svm_poly_fit, train_data)
train_error_poly <- mean(pred_train_poly != train_data$Purchase)

pred_test_poly <- predict(svm_poly_fit, test_data)
test_error_poly <- mean(pred_test_poly != test_data$Purchase)

train_error_poly #0.3725
test_error_poly #0.4407407

tune.out.poly <- tune(svm, Purchase ~ ., data = train_data, kernel = "polynomial", degree = 2, ranges = list(cost = c(0.01, 0.1, 1, 10)))
summary(tune.out.poly)

best_cost_poly <- tune.out.poly$best.parameters$cost
best_cost_poly


svm_poly_best <- svm(Purchase ~ ., data = train_data, kernel = "polynomial", degree = 2, cost = best_cost_poly)
train_best_poly_preds <- predict(svm_poly_best, train_data)
test_best_poly_preds <- predict(svm_poly_best, test_data)

train_error_best_poly <- mean(train_data$Purchase != train_best_poly_preds)
test_error_best_poly <- mean(test_data$Purchase != test_best_poly_preds)

print(train_error_best_poly)
# 0.15
print(test_error_best_poly)
# 0.1962963


#### H ####

#best model is the one that has the lowest test error
#tuning with best cost helped bring down error a lot for radial and polynomial

#linear best cost test error: 0.1814815

#radial best cost test error: 0.1925926

#polynomial best cost test error: 0.1962963

#all similiar, and linear is best, linear is most interpretable, so choose the linear model
#decision boundary is a linear function of the input features


