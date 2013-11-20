import pandas as pd
import numpy as np

def readTrainingData(fn = 'spline_train3000.csv'):
    return pd.read_csv(fn,header=None)

def calculateCoefficients():
	#epsilon = sqrt(6)/sqrt(Lin + Lout)
	Lin = 14
	hidden = 10
	Lout = 1
	epsilon_init = math.sqrt(6)/math.sqrt(Lin + hidden)
	theta1 = np.random.rand(hidden,1 + Lin) * 2 * epsilon_init - epsilon_init
	theta2 = np.random.rand(Lout,1 + hidden) * 2 * epsilon_init - epsilon_init
	return theta1, theta2

	
def normalizeXy(data):
	data = readTrainingData()
	data = data.loc[:,2:]
	y = data.iloc[:,-1].T
	X = data.iloc[:,:-1]
	normX = (X-X.mean())/X.std()
	normy = y - y.mean()
	return normX, normy
	
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
	Description: Implements backward propagation algorithm
	Parameters:
	normX, normy - normalized variables containing training data
	W1, W2 - Initial coefficients
'''
def nnCoefficients(normX, normy, W1, W2)	
	Delta1 = np.zeros((W1.shape))
	Delta2 = np.zeros((W2.shape))
	#Neural networks
	for i in range(X.shape[0]):
		a1 = normX.iloc[i,:].values
		a1 = np.insert(a1,0,1) 
		a1.shape=(1,len(a1)) # change shape to (1, 15)
		z2 = np.dot(a1, W1.T)
		a2 = sigmoid(z2)  
		a2 = np.insert(a2,0,1) # a2.shape - (1, 11)
		z3 = np.dot(a2, W2.T)
		a3 = sigmoid(z3)  # a3.shape = (1, 1)
		delta3 = a3 - normy[i] # delta3.shape = (1,1)
		delta2 = delta3 * W2 * a2 * (1 - a2) # delta2.shape = (1,11)
		Delta2 = Delta2 + delta3 * a2
		Delta1 = Delta1 + np.dot(delta2.T[1:], a1)
	Theta1_grad = Delta1 / X.shape[0]
	Theta2_grad = Delta2 / X.shape[0]