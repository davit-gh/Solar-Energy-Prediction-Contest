'''
Created on 2013-8-21

@author: Davit Kartashyan
@email : davsmile@yahoo.com
'''
import numpy as np
import scipy.io as sio

'''
    createSubmission()
    Description: Creates submission-ready csv file based on theta coefficients
                    calculated by Octave
'''
def createSubmission():
    theta=sio.loadmat('octave_thetas.txt')
    thetas=theta['theta']
    thetas.shape=(16,)
    mn=theta['mn']
    sigma=theta['sigma']
    f = np.genfromtxt('spline_test.csv',delimiter=',',usecols=range(1,16))
    normf=(f-mn)/sigma
    normf=np.hstack((np.ones((len(normf),1)),normf))
    pred=normf*thetas
    pred=np.sum(pred,axis=1)
    final=np.reshape(pred,(1796,-1))
    ff=open('final_sub.csv','w')
    date=np.genfromtxt('sampleSubmission.csv',delimiter=',',skip_header=1,dtype=int)[:,0]
    date.shape=(date.shape[0],1)
    f=open('sampleSubmission.csv').readline()
    finalwithdate=np.hstack((date,final))
    ff.write(f)
    np.savetxt(ff,finalwithdate,delimiter=',',fmt='%i')
    ff.close()