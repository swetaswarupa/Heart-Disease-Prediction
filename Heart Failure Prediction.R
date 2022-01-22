if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(gglot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(pscl)) install.packages("pscl", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(randomForest)
library(ggplot2)
library(corrplot)
library(dplyr)
library(GGally)


#Reading the data
heart <- read.csv('C:/Heart Failure Prediction/heart.csv', header = TRUE)
str(heart)

#There are 918 observations and 12 variables. There are 5 Character variables, 6 integer variables and 1 numeric variable.

#Checking the number of unique values for each variable.
sapply(heart, n_distinct)

#Features
#Age: age of the patient [years]
#Sex: sex of the patient [M: Male, F: Female]
#ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
#RestingBP: resting blood pressure [mm Hg]
#Cholesterol: serum cholesterol [mm/dl]
#FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
#RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
#MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
#ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
#Oldpeak: oldpeak = ST [Numeric value measured in depression]
#ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
#HeartDisease: output class [1: heart disease, 0: Normal]

#Lets look at the dataset
head(heart)

#Checking for any Missing Values.
sapply(heart, function(x) sum(is.na(x)))

#There are no missing values.

#Checking for target imbalance
heart$HeartDisease <- as.character(heart$HeartDisease)
ggplot(data = heart, aes(x=HeartDisease, fill = HeartDisease)) +
  geom_bar() + labs(x='Heart Disease') + labs(title = "Bar Graph of Heart Disease")
  
table(heart$HeartDisease)
prop.table(table(heart$HeartDisease))

# The target looks balanced.

#Data Analysis

#Plotting Sex Variable
ggplot(data = heart, aes(x=Sex, fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='Sex') + labs(title = "Bar Graph of Heart Disease by Sex")

#We can say that males are affected more by heart disease than females.


#Plotting Age Variable
ggplot(data = heart, aes(x=Age, fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='Age') + labs(title = "Bar Graph of Heart Disease by Age")

#Plotting ChestPainType Variable
ggplot(data = heart, aes(x=reorder(ChestPainType, ChestPainType, function(x)-length(x)), fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='Chest Pain Type') + labs(title = "Bar Graph of Heart Disease by Chest Pain Type")

#proportion of heart patients is higher when the chest pain type is ASY.

#Plotting RestingECG Variable
ggplot(data = heart, aes(x=reorder(RestingECG, RestingECG, function(x)-length(x)), fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='Resting ECG') + labs(title = "Bar Graph of Heart Disease by Resting ECG")

#proportion of heart patients is higher when the Resting ECG type is LVH and ST.

#Plotting ExerciseAngina Variable
ggplot(data = heart, aes(x=ExerciseAngina, fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='Exercise Angina') + labs(title = "Bar Graph of Heart Disease by Exercise Angina")

#proportion of heart patients is significantly higher when the ExerciseAngina is Y.

#Plotting ST_Slope Variable
ggplot(data = heart, aes(x=reorder(ST_Slope, ST_Slope, function(x)-length(x)), fill = HeartDisease, position = 'dodge')) +
  geom_bar() + labs(x='ST Slope') + labs(title = "Bar Graph of Heart Disease by ST Slope")

#proportion of heart patients is significantly higher when the ST_Slope is Flat and Down.


#Pair Plot
heart1 <- heart %>% select(-c(Sex,ChestPainType,RestingECG,ExerciseAngina,ST_Slope))
heart1$HeartDisease <- as.character(heart1$HeartDisease) 
ggpairs(heart1, ggplot2::aes(colour=HeartDisease)) 


#Checking correlation between variables
library(corrplot)
#Converting Categorical Variables into Numerical Variable
heart2 <- heart %>% mutate_if(is.character, as.factor)
heart2 <- heart2 %>%  mutate_if(is.factor, as.numeric)

corrplot(cor(heart2), type="full", 
         method ="color", title = "Heart correlatoin plot", 
         mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="indianred4")
round(cor(heart2),2)


# Splitting the Data into training and test data sets
library(caret)
heart <- heart %>% mutate_if(is.character, as.factor)
heart$HeartDisease <- as.factor(heart$HeartDisease)
set.seed(5)
trainIndex <- createDataPartition(heart$HeartDisease, p = .7,
                                  list = FALSE,
                                  times = 1)
Train <- heart[ trainIndex,]
Test <- heart[-trainIndex,]

prop.table(table(Train$HeartDisease))
prop.table(table(Test$HeartDisease))

#Model 1 Logistic Regression

lm <- glm(HeartDisease ~.,family=binomial(link='logit'),data=Train)

anova(lm, test="Chisq")

library(pscl)
pR2(lm)

pred_lm <- predict(lm, newdata = Test, type = 'response')
pred_lm <- ifelse(pred_lm > 0.5,1,0)
misClasificError <- mean(pred_lm != Test$HeartDisease)
print(paste('Accuracy',1-misClasificError))

library(ROCR)
p <- predict(lm, newdata=Test, type="response")
pr <- prediction(p, Test$HeartDisease)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


#Model 2 Random Forest Model
library(randomForest)
set.seed(5)
rf <- randomForest(HeartDisease~.,data = Train, type = "class", importance=TRUE, ntree= 500, mtry = 3)
print(rf)
plot(rf)


varImpPlot(rf, main ='Feature Importance')

pred_rf <- predict(rf, Test, type = "class")

confusionMatrix(as.factor(pred_rf),as.factor(Test$HeartDisease))

#Model 3 Naive Bayes Model
# set up 10-fold cross validation procedure
library(caret)
train_control <- trainControl(
  method = "cv", 
  number = 10
)

# train model
set.seed(123)
nb <- train(
  x = Train,
  y = Train$HeartDisease,
  method = "nb",
  trControl = train_control
)
nb

confusionMatrix(nb)

pred_nb <- predict(nb, newdata = Test)
confusionMatrix(pred_nb, Test$HeartDisease)