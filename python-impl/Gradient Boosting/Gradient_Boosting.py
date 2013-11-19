from numpy import *
'''
	stumpReg(dataSet,dim,thresh,ineq)
	Description: Creates a simple stump by taking mean values of target 
			variable(classLabel) in each of 2 branches
	Parameters: 
	dataSet - training data
	classLabel - target for the prediction (dependent variable)
	dim - dimension of the feature vector
	thresh - threshold value
	ineq - inequality('less than', 'greater than') 
	Returns:
	retArr - the resulting array after splitting
	select - boolean array that defines values in 2 branches
'''
def stumpReg(dataSet,dim,thresh,ineq):
    retArr = ones((dataSet.shape[0],1))
    if ineq == 'lt':
        select = dataSet[:,dim] <= thresh
        retArr[select] = mean(dataSet[:,-1][select])
        retArr[~select] = mean(dataSet[:,-1][~select])
    else:
        select = dataSet[:,dim] > thresh
        retArr[select] = mean(dataSet[:,-1][~select])
        retArr[~select] = mean(dataSet[:,-1][select])
    return retArr, select

'''
	buildStumpReg(dataSet,classLabel)
	Description: A simple stump. Iterates over dataset's all columns and 
				 finds the best split which results in the most error reduction 
	Parameters:
	dataSet(list): feature list
	classLabel(list): target variable list
'''
def buildStumpReg(dataSet,classLabel):
    dataSet = mat(dataSet); classLabel = mat(classLabel).T
    m,n = shape(dataSet)
    stepNum = 10.0; bestClassEst = mat(zeros((m,1))); bestStump = {}
    minError = inf
    for i in range(n):
        minRange = dataSet[:,i].min(); maxRange = dataSet[:,i].max()
        stepSize = (maxRange - minRange) / stepNum
        for j in range(int(stepNum)):
            for ineq in ['lt','gt']:
                thresh = (minRange + float(j) * stepSize)
                classArr, selected = stumpReg(dataSet,i,thresh,ineq) 
                errArr = err(classLabel,classArr)
                totalErr = errArr.sum()
                if totalErr < minError:
                    minError = totalErr
                    bestClassEst = classArr.copy()
                    bestSelect = selected.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = thresh
                    bestStump['ineq'] = ineq
    return bestStump, minError, bestClassEst, bestSelect
	
'''
	err(actual,prediction)
	Description: Absolute error
'''
def err(actual,predicted):
    return abs(actual-predicted)
	
'''
	Description: Find optimal gamma gamma so that the sum of residuals is minimum
	Parameters:
	sub : Difference between predicted and actual values of y
	yhat: estimated value of y
	y   : actual y
'''
def findMinGamma(sub,yhat,y):
    gamma = inf
    for error in [sub.max(), sub.min(), sub.mean()]:
        if err(yhat+gamma,y).sum() > err(yhat+error,y).sum():
            gamma = error
    return gamma

'''
	TreeBoost(dataset, y, numIt=19)
	Description: This is the actual implementation of Friedman's Gradient 
				 Boosting Machine using decision tree as a base learner
	Parameters:
	dataset(list): training features
	y(list): target variable
	numit: number of iterations
'''
def TreeBoost(dataset, y, numIt=19):
    N, numFeat = dataset.shape # N is the number of training entries
    weakPredictors = []
    stump, error, classEst,sel = buildStumpReg(dataset,y.T)
    weakPredictors.append(classEst.T)
    gradLoss = zeros((N,1))
    for m in range(numIt):
        gradLoss = sign(y.T - classEst) #gradient of the absolute error loss function
        bestFittedStump,fittedError,f,selected = buildStumpReg(dataset,gradLoss) # fitting a tree to target *gradLoss*
        f=mat(f)
        left = f[selected]
        yLeft = gradLoss[selected]
        right = f[~selected]
        yRight = gradLoss[~selected]
        subLeft=yLeft-left
        subRight=yRight-right
        gammaLeft = findMinGamma(subLeft,left,yLeft)
        gammaRight = findMinGamma(subRight,right,yRight)
        gamma = selected*gammaLeft + ~selected*gammaRight
        bestFittedStump['gamma'] = gamma
        weakPredictors.append(bestFittedStump)
        f += multiply(f,gamma)
    return f,weakPredictors

'''
	gradBoostPredict(testData, weakPredictors)
	Description: Making predictions on test dataset
	Parameters:
	testData - Test dataset
	weakPredictors - List of weak predictors obtained from TreeBoost() function
'''
def gradBoostPredict(testData, weakPredictors):
    testData = mat(testData)
    m = testData.shape[0]
    pred = mat(zeros((m,1)))
    classEst = weakPredictors[0]
    for i in range(1,len(weakPredictors)):
        classEst, select = stumpReg(testData,classEst, weakPredictors[i]['dim'], \
                                    weakPredictors[i]['thresh'], weakPredictors[i]['ineq'])
        classEst += multiply(classEst,weakPredictors[i]['gamma'])
        classEst = classEst.T
    return pred