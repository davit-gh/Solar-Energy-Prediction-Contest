'''
Created on 2013-8-17

@author: Davit Kartashyan
@email : davsmile@yahoo.com
'''
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn import grid_search
from sklearn import cross_validation

SEED = 11

#Loading data and obtaining features and outcome   
def loadRealData(fname):
    data = np.genfromtxt(fname,delimiter=',', usecols=range(1,17))
    features = data[:,:-1]
    outcome = data[:,-1]
    outcome.shape = (outcome.shape[0],1)
    return features,outcome

'''linear regression with feature standadrdization
(using Normal Equation for finding optimal parameters)
'''
def regression(feat,y):
    feat = np.mat(feat)
    meanfeat = np.mean(feat,0)
    var = np.var(feat,0)
    meany = np.mean(y)
    feat = (feat - meanfeat) / var
    feat = np.hstack((np.ones((feat.shape[0],1)),feat))
    y = y - meany
    xTx = feat.T*feat
    if np.linalg.det(xTx) == 0:
        print "No solution"
        return
    w = xTx.I * (feat.T*y)
    return w, meanfeat, meany, var

''' Ridge regression with lambda = 0.2
	(using Normal Equation for finding optimal parameters)
'''
def ridgeRegression(feat,y,lam = 0.2):
    xTx = feat.T * feat
    denom = xTx + np.eye(feat.shape[1]) * lam
    if np.linalg.det(denom) == 0:
        print "No solution"
        return
    regr = denom.I * (feat.T * y)
    return regr
	
''' Description: normalizes input variables and performs cross - validation
				 for Ridge regression
	Parameters: training and cross-validation datasets(numpy.narray)
'''	
def cvRidge(trainfeat,cvfeat,trainy,cvy):
    feat = np.mat(trainfeat); y = np.mat(trainy)
    mnfeat = np.mean(feat,0)
    mny = np.mean(y)
    var = np.var(feat,0)
    normfeat = (feat - mnfeat) / var 
    normfeat = np.hstack((np.ones((normfeat.shape[0],1)),normfeat))
    normy = y - mny
    numTest = 30
#    ws = np.zeros((numTest,feat.shape[1]))
    normcvfeat = (cvfeat - mnfeat) / var
    normcvfeat = np.hstack((np.ones((normcvfeat.shape[0],1)),normcvfeat))
    normcvy = cvy - mny
    lambdas = []
    maes = []
    for i in range(numTest):
        w = ridgeRegression(normfeat,normy,np.exp(i-25))
        pred = normcvfeat * w
        mae = metrics.mean_absolute_error(pred,normcvy)
        print "%d/%d mae: %f" % (i,numTest,mae)
        maes.append(mae)
    print "lambda for min mae exp(%d-25): %f" % (np.argmin(maes),np.exp(np.argmin(maes)-25))
    return np.min(maes), np.exp(np.argmin(maes)-25)


''' Description: Standardizing cv dataset using 
	mean and variance of the training dataset
	Parameters: training and cross-validation datasets(numpy.narray)
'''
def cvLinearRegression1(trainfeat,cvfeat,trainy,cvy):
    thetas, meanfeat, meany, var = regression(trainfeat,trainy)
    normfeat = (cvfeat - meanfeat) / var
    normfeat = np.hstack((np.ones((normfeat.shape[0],1)),normfeat))
    normcvy = cvy - meany
    hx = normfeat * thetas
    return hx, normcvy
#    print np.sum(np.abs(hx[0]-normy))/normy.shape[0]
#    print metrics.mean_absolute_error(hx,normcvy)

''' Description: Plotting fitted line for a fixed lambda
	Parameters: Arrays of features(feat) and target variables(y)
'''
def plotRidgeLambdas(feat,y):
    ws = testRidge(feat,y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ws)
    plt.xlim([-10,20])
    plt.show()

''' Description: Performs n-fold cross-validation
	Parameters: 
	feat : array of features
	y    : target variables
	model: model to be used for cv
	n    : number of folds
'''
def nfoldcv(feat,y,model,n=10):
    MAE = 0.0
    minls = 0.0
    for i in range(n):
        trainX,cvX,trainY,cvY = cross_validation.train_test_split\
        (feat,y,test_size=.3,random_state=i*SEED)
        #model.fit(trainX,trainY)
        #pred = model.predict(cvX)
 #      pred,normcvY = testRegression1(trainX,cvX,trainY,cvY)
 #      mae = metrics.mean_absolute_error(pred,normcvY)
        mae,minlambda = testRidge(trainX,cvX,trainY,cvY)
        print "mae for %d/%d run is %f" % (i,n,mae)
        MAE += mae
        minls += minlambda
    print "Average mae: %f\nAverage lambda: %f" % (MAE/float(n),minls/float(n))


''' main() function uses Ridge() function from 
	sklearn.linear_model package and nfolcv() 
	for cross-validation
'''
def main():
    feat,y = loadRealData('spline_train.csv')
    model = Ridge(normalize=True)
#   parameter = {'alpha':np.logspace(-6,6,10,base=10)}
#   grid_search.GridSearchCV(model,parameter)
#   scores = cross_validation.cross_val_score(model, feat, y, cv=5, scoring=metrics.mean_absolute_error)
#   model.fit(feat, y)
    nfoldcv(feat,y,model)
    


if __name__ == '__main__':
    main()v
