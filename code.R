rm(list=ls())
#Impulse
library(mlr)
library(caret)
train<-read.csv("Train.csv")
# summary<-summarizeColumns(train)
 train$label<-as.factor(train$label)

library(dplyr)
x<-summarizeColumns(train1)%>%filter(max==0)
x<-x$name
plot(train$label)

train1<-train[,2:785]
train2<-subset(train1,select = -which(names(train1)%in%x))
train3<-cbind(train$label,train2)
colnames(train3)[1]<-"label"

# label1<-train%>%filter(label==1)
# label2<-train%>%filter(label==2)
# label3<-train%>%filter(label==3)
# label4<-train%>%filter(label==4)
# label5<-train%>%filter(label==5)
# label6<-train%>%filter(label==6)
# label7<-train%>%filter(label==7)
# label8<-train%>%filter(label==8)
# label9<-train%>%filter(label==9)
# label0<-train%>%filter(label==0)

library(xgboost)
set.seed(717)
# Make split index
train_index <- sample(1:nrow(train3), nrow(train3)*0.75)
# Full data set
data_variables <- as.matrix(train3[,-1])
data_label <- train3[,"label"]
data_matrix <- xgb.DMatrix(data = as.matrix(train3), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)
numberOfClasses <- length(unique(train3$label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses
                   )
                  
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5
# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                    data = train_matrix, 
                    nrounds = nround,
                    nfold = cv.nfold,
                    verbose = TRUE,
                    prediction = TRUE)
cv_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label+1)
head(cv_prediction)
confusionMatrix(factor(cv_prediction$label), 
                factor(cv_prediction$max_prob),
                mode = "everything")
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)
# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")
# get the feature real names
names <-  colnames(train3[,-1])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)
# plot
gp = xgb.plot.importance(importance_matrix[1:10])
print(gp)


#Test Data
data.test<-read.csv("Test_Data.csv")
summarizeColumns(data.test)
x.test<-summarizeColumns(data.test)%>%filter(max==0)

table(x.test$name%in%x)
x.test1<-x.test%>%filter(x.test$name%in%x==TRUE)

x%in%x.test1$name
x[66]
x[63]

data.test2<-subset(data.test,select = -which(names(data.test)%in%x))
data.test2$label<-sample(0:9,size=10000,replace=T)
test_variables1 <- as.matrix(data.test2[,-718])
test_label1 <- data.test2[,"label"]
test_matrix1 <- xgb.DMatrix(data = as.matrix(data.test2), label = test_label1)

#prediction on the test set
test_pred1 <- predict(bst_model, newdata = test_matrix1)
test_prediction1<- matrix(test_pred1, nrow = numberOfClasses,
                          ncol=length(test_pred1)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(max_prob = max.col(., "last")-1)

colnames(test_prediction1)[11]<-"label"


write.csv(test_prediction1$label,"Submission.csv")





