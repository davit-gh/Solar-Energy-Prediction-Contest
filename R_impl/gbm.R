gbmTrain <- function(Xtrain = 'x_train.csv', ytrain = 'y_train.csv',
					Xtest = 'x_test.csv', ytest = 'y_test.csv'){
	x_train<-read.csv(Xtrain,header=F)
	x_test<-read.csv(Xtest,header=F)
	y_test<-read.csv(ytest,header=F)
	y_train<-read.csv(ytrain,header=F)
	model<-gbm(y_train$V1 ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13, 
		data=x_train,distribution="gaussian", shrinkage=0.01,
		interaction.depth=2, n.minobsinnode=1,n.trees=300)
	pred<-predict.gbm(model,x_test,n.trees=300)
	mae(y_test,pred)
}