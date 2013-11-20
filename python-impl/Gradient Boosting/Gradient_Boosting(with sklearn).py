from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import pylab as pl
import numpy as np
from sklearn import datasets

'''
	GradBoost(X,y)
	Description: Performs gradient boosting and outputs plots
	Parameters :
	X - Training features
	y - Target variable
'''
def GradBoost(X,y):
    #load dataset
    #boston = datasets.load_boston()
   # X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    offset = int(X.shape[0]*0.7)
    train_X, train_y = X[:offset], y[:offset]
    test_X, test_y = X[offset:], y[offset:]
    
    #model building
    params = {'n_estimators':1500, 'max_depth':4, 'min_samples_split':2, 'learning_rate':0.1, 'loss':'ls' }
    cls = GradientBoostingRegressor(**params)
    cls.fit(train_X,train_y)
    MAE = mean_absolute_error(test_y,cls.predict(test_X))
  #  print "MSE: %f"%MSE
    
    #plot trianing deviance
    test_score = np.zeros((params['n_estimators'],),dtype=np.float64)
    for i,pred in enumerate(cls.staged_decision_function(test_X)):
        test_score[i] = cls.loss_(test_y,pred)
    pl.figure(figsize=(12,6))
    pl.subplot(1,2,1)
    pl.plot(np.arange(params['n_estimators'])+1, cls.train_score_, 'b-', label='Training set Deviance')
    pl.plot(np.arange(params['n_estimators'])+1, test_score, 'r-', label='Test set Deviance')
    
    # plot feature importance
    feature_importance = cls.feature_importances_
    feature_importance = (feature_importance / feature_importance.max()) * 100.0
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1,2,2)
    pl.barh(pos, feature_importance[sorted_idx],align='center')
    #pl.yticks(pos,boston.feature_names[sorted_idx])
    pl.xlabel("Relative importance")
    pl.title("Variable importance")
    pl.show()
    return MAE, cls

'''
	makePrediction(test_file, sampleSubmission_file)
	Description: Makes preiction on test dataset
	Parameters : Input files
'''
def makePrediction(test_file = "spline_test.csv",sampleSubmission_file = "sampleSubmission.csv'"):
	test = np.genfromtxt(test_file,delimiter=',')
	predict=cls.predict(test1)
	predict1=predict
	pr_reshaped=np.reshape(predict1,(-1,98))
	ff=open('submission_131026_scikit_GBR1.csv','w')
	date=np.genfromtxt(sampleSubmission_file, delimiter=',',skip_header=1,dtype=int)[:,0]
	date.shape=(date.shape[0],1)
	f=open(sampleSubmission_file).readline()
	finalwithdate=np.hstack((date,pr_reshaped))
	ff.write(f)
	np.savetxt(ff,finalwithdate,delimiter=',',fmt='%i')
	ff.close()
	
if __name__ == '__main__':
    mae, cls = GradBoost(X,y)
	print mae #Out: 1841064.8366720795
	makePrediction()