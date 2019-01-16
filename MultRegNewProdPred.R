### Riley Barrow
### Course 2 // Task 3
### Multiple Regression
### New Product Prediction 

install.packages("readr")
library(readr)

library(caret)
install.packages("lattice")
install.packages("ggplot2")

## Pre-Process the Data ##

#assign objects
existing_products <- existingproductattributes2017
new_products <- newproductattributes2017

#dummify the data, creates ProductType binary codes to separate products 
#and uniquely identify them 
newDataFrame <- dummyVars(" ~ .", data = existing_products)
readyData <- data.frame(predict(newDataFrame, newdata = existing_products))

str(existing_products)
summary(existing_products)
names(readyData)

write.csv(readyData, "readyData.csv")

#delete attributes with missing data (best sellers rank)
readyData $ BestSellersRank <- NULL
readyData
names(readyData)

#delete attributes not needed to look at
readyData$ProductTypeAccessories <- NULL
readyData$ProductTypeDisplay <- NULL
readyData$ProductTypeExtendedWarranty <- NULL
readyData$ProductTypeGameConsole <- NULL
readyData$ProductTypePrinter <- NULL
readyData$ProductTypePrinterSupplies <- NULL
readyData$ProductTypeSoftware   <- NULL
readyData$ProductTypeTablet <- NULL
names(readyData)
View(readyData)

#feature engineering
#find correlation
corrData <- cor(readyData)
corrData
View(corrData)  #exported to excel 

#visualize corr matirx
install.packages("corrplot")
library(corrplot)
corrplot(corrData)

#remove high correlation attributes
readyData $ x2StarReviews <- NULL
readyData $ x3StarReviews <- NULL
readyData $ x5StarReviews <- NULL
View(readyData)

#export to visulaize 
write.csv(readyData, "readyDataFiltered.csv")
write.csv(corrData, "corrDataFiltered.csv")

## Model Training and Tuning ##

set.seed(2727)

#define train/test split of Data // 75% / 25%
inTraining <- createDataPartition(readyData $ Volume, p = 0.75, list = FALSE)
training <- readyData[inTraining,]
testing <- readyData[-inTraining,]

## Random Forest

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#run a linear regression model
set.seed(2727)
lm1 <- train(Volume ~., data = training, 
             method = "lm",
             preProc = c("center", "scale"),
             trControl = fitControl)
lm1

varImp(lm1)
varImpPlot(lm1)

#evaluate linear regression model
lmVolumePredict <- predict(lm1, testing)
postResample(pred = lmVolumePredict, obs = testing$Volume)
xyplot(lmVolumePredict ~ testing$Volume)


library(randomForest)

#train RF regression model, tuneLength = 1
set.seed(2727)
rfFit1 <- train(Volume~., data = training,
                method = "rf",
                trControl = fitControl,
                preProc = c("center", "scale"),
                tuneLength = 3)  #adjust tuneLength and compare results
rfFit1
plot(rfFit1)
varImp(rfFit1)
summary(rfFit1)

#train RF model with grid
set.seed(2727)
fitControl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")
rfGrid <- expand.grid(mtry = seq(1,15, by = 1))
rfFit2 <- train(Volume~., data = training, 
                method = "rf",
                trControl = fitControl1,
                preProc = c("center", "scale"),
                tuneGrid = rfGrid,
                tuneLength = 3)
rfFit2
plot(rfFit2)
varImp(rfFit2)

#evaluate RF model performance 
rfVolumePredict1 <- predict(rfFit1, testing)
postResample(pred = rfVolumePredict1, obs = testing$Volume)
xyplot(rfVolumePredict1 ~ testing$Volume)
plot(rfVolumePredict1) 

rfVolumePredict2 <- predict(rfFit2, testing)
postResample(pred = rfVolumePredict2, obs = testing$Volume)
xyplot(rfVolumePredict2 ~ testing$Volume)


## SVM model ##
#SVM Radial model
set.seed(2727)
svmFit1 <- train(Volume~.,
                 data = training,
                 method = "svmRadial",
                 preProcess = c("BoxCox", "center", "scale"),
                 trControl = fitControl,
                 tuneLength = 3)

svmFit1

#SVM poly mofdel
set.seed(2727)
svmFit2 <- train(Volume~.,
                 data = training,
                 method = "svmPoly",
                 preProcess = c("BoxCox", "center", "scale"),
                 trControl = fitControl,
                 tuneLength = 3)
svmFit2

#evaluate the SVM model
svmVolumePredict1 <- predict(svmFit1, testing)
postResample(pred = svmVolumePredict1, obs = testing$Volume)
xyplot(svmVolumePredict1 ~ testing$Volume)


svmVolumePredict2 <- predict(svmFit2, testing)
postResample(pred = svmVolumePredict2, obs = testing$Volume)

## Gradient Boost Model ##

#GBM Tuning Parameters: 
  # n.trees (# of boosting iterations)
  # interaction.depth (max tree depth)
  # shrinkage
  # n.minobsinnode (min terminal node size)

set.seed(2727)
metric <- "RMSE"
gbmGrid <- expand.grid(mtry = c(1:15))
gbmFit1Control <- trainControl(method = "cv", number = 10) #repeats = 3)
#gbm tuneLength
gbmFit1 <- train(Volume~., data = training,
                 distribution = "gaussian",
                 method = "gbm",
                 trControl = gbmFit1Control,
                 verbose = FALSE,
                 tuneLength = 1)
gbmFit1

#gbmFit2
set.seed(2727)
gbmFit2 <- train(Volume~., data = training,
                 distribution = "gaussian",
                 method = "gbm",
                 trControl = gbmFit1Control,
                 verbose = FALSE,
                 bag.fraction = 0.75)
gbmFit2

#evalute model performance 
gbmVolumePredict1 <- predict(gbmFit1, testing)
postResample(pred = gbmVolumePredict1, obs = testing$Volume)

gbmVolumePredict2 <- predict(gbmFit2, testing)
postResample(pred = gbmVolumePredict2, obs = testing$Volume)
xyplot(gbmVolumePredict2 ~ testing$Volume)
#evaluate results
results <- resamples(list(rf = rfFit1, svm = svmFit1, gbm = gbmFit1))
summary(results)

## NEW PRODUCT DATA SET ##
##pre-processing new porducts data set 
newDataFrame2 <- dummyVars("~.", data = new_products)
readyData2 <- data.frame(predict(newDataFrame2, newdata = new_products))

names(readyData2)
write.csv(readyData2, "readyData2.csv")

#delete attributes with missing data (best sellers rank)
readyData2 $ BestSellersRank <- NULL
names(readyData2)

#delete attributes not needed to look at
readyData2$ProductTypeAccessories <- NULL
readyData2$ProductTypeDisplay <- NULL
readyData2$ProductTypeExtendedWarranty <- NULL
readyData2$ProductTypeGameConsole <- NULL
readyData2$ProductTypePrinter <- NULL
readyData2$ProductTypePrinterSupplies <- NULL
readyData2$ProductTypeSoftware   <- NULL 
readyData2$ProductTypeTablet <- NULL
names(readyData2)

#remove high correlation attributes
readyData2 $ x2StarReviews <- NULL
readyData2 $ x3StarReviews <- NULL
readyData2 $ x5StarReviews <- NULL
View(readyData2)

write.csv(readyData2, "readyData2X.csv")

#make final predictions
finalPred <- predict(rfFit1, readyData2)
finalPred
View(finalPred)

#export Final pred formatted on new products data sheet 
output <- new_products 
output$predictions <- finalPred

write.csv(output, file="FINALPREDrfFit1.csv", row.names = TRUE)

