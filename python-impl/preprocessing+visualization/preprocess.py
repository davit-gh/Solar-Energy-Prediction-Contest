'''
Created on 2013-8-12

@author: Davit Kartashyan
@email : davsmile@yahoo.com
'''
"""
Modified version of solarSplines.py
Modified by: Davit Kartashyan
Authors: David John Gagne II, Andrew MacKenzie, Darrel Kingfield, Ben Herzog, Tim Humphrey, Thibault Lucidarme
Description: Reads Mesonet and GEFS data and applies Catmull-Rom splines to interpolate GEFS data to Mesonet sites.
			 Makes a 15-column dataset where each column corresponds to a provided forecast variable.
			 For each row in 'train.csv' corresponds 98 rows in the converted dataset.
"""
from netCDF4 import Dataset
import numpy as np
import os
from copy import deepcopy
import math
def loadMesonetData(filename, stationFilename="E:/kaggle/Solar/gefs_train/train/station_info.csv"):
    """
    loadMesonetData(filename,stationFilename)
    Description: loads Mesonet data and station data
    Parameters:
    filename (str) - Name of Mesonet csv file being read.
    stationData (str) - Name of file containing station information. Default station_info.csv.
    Returns:
    data - numpy array containing total daily solar radiation for each date
    dates - numpy array of dates as integers in YYYYMMDD format
    stationData - numpy structured array containing the station information, including lat-lon and elevation.
    """
    data = np.genfromtxt(filename,delimiter=',',skiprows=1,dtype=float)
    dates_test = np.array(data[:,0].T,dtype=int)
    data = data[:,1:]
    stationData = np.genfromtxt(stationFilename,delimiter=',',dtype=[("stid","S4"),("nlat",float),("elon",float),("elev",float)],skiprows=1)
    return data,dates_test,stationData
def loadFromTrain(filename_train):
    train = np.genfromtxt(filename_train,delimiter=',',skiprows=1,dtype=float)
#    dates_train = np.array(train[:,0].T,dtype=int)
#    ind = np.where(date == dates_train)[0]
    return train

def loadTrainDates(train):
    return np.array(train[:,0].T, dtype = int)

def dataForDate(train,date):
    return [line[1:] for line in train if line[0] == date][0]

def loadData(filename):
    """
    loadData()
    Description: Creates a netCDF4 file object for the specified file.
    Parameters:
    filename (str) - name of the GEFS netCDF4 file.
    Returns:
    data - Dataset object that allows access of GEFS data.
    """
    data = Dataset(filename)
    return data
#def loadAllFiles
def getGrid(data,date,fHour,eMember):
    """
    getGrid()
    Description: Load GEFS data from a specified date, forecast hour, and ensemble member.
    Parameters:
    data (Dataset) - Dataset object from loadData
    date (int) - date of model run in YYYYMMDD format
    fHour (int) - forecast hour
    eMember (int) - ensemble member id.
    Returns: numpy 2d array from the specified output
    """
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] == fHour)[0]
    eIdx = np.where(data.variables['ens'][:] == eMember)[0]
    return data.variables.values()[-1][dateIdx,eIdx,fIdx][0]

def getDailyMeanSumGrid(data,date):
    """
    getDailyMeanSumGrid()
    Description: For a particular date, sums over all forecast hours for each ensemble member then takes the 
    mean of the summed data and scales it by the GEFS time step.
    Parameters:
    data (Dataset) - netCDF4 object from loadData
    date (int) - date of model run in YYYYMMDD format
    Returns - numpy 2d array from the specified output
    """
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] <= 24)[0]
    s = data.variables.values()[-1][dateIdx,:,fIdx,:,:]
    ret = s.sum(axis=2)
    retmean = ret.mean(axis=1)[0]
    return retmean

def calcDistance(lat1, lon1, lat2, lon2):
    R = 6371 #km
    d = math.acos(math.sin(lat1)*math.sin(lat2) + 
                  math.cos(lat1)*math.cos(lat2) *
                  math.cos(lon2-lon1)) * R;
    return d
    
def buildSplines(data,grid,stationdata):
    """
    buildSplines()
    Description: For each station in stationdata, a set of Catmull-Rom splines are calculated to interpolate from the
    nearest grid points to the station location. A set of horizontal splines are created at each latitude and 
    interpolated at the station longitude. Then another spline is built from the output of those splines to get the 
    value at the station location.
    Paramters:
    data (Dataset) - netCDF4 object with the GEFS data 
    grid (numpy array) - the grid being interpolated
    stationdata (numpy structured array) - array containing station names, lats, and lons.
    Returns: array with the interpolated values.
    """
    
    outdata=np.zeros(stationdata.shape[0])
    print stationdata.shape
    for i in xrange(stationdata.shape[0]):
        slat,slon=stationdata['nlat'][i],stationdata['elon'][i]
        nearlat=np.where(np.abs(data.variables['lat'][:]-slat)<1)[0]
        print np.array(data.variables['lat'][:],data.variables['lon'][:])
        print "huh?"
        nearlon=np.where(np.abs(data.variables['lon'][:]-slon-360)<1)[0] 
        Spline1=np.zeros(nearlon.shape)
        for l,lat in enumerate(nearlat):
            Spline1[l]=Spline(grid[nearlat[l],nearlon],(slon-np.floor(slon))/1)
        outdata[i]=Spline(Spline1,(slat-np.floor(slat))/1)
    return outdata

def Spline(y,xi):
    """
    Spline
    Description: Given 4 y values and a xi point, calculate the value at the xi point.
    Parameters:
    y - numpy array with 4 values from the 4 nearest grid points
    xi - index at which the interpolation is occurring.
    Returns: yi - the interpolated value
    """
    return 0.5*((2*y[1])+(y[2]-y[0])*xi+(-y[3]+4*y[2]-5*y[1]+2*y[0])*xi**2+(y[3]-3*y[2]+3*y[1]-y[0])*xi**3)

'''
	Description: Uses the whole training dataset(Takes long time to finish)
'''
def main1(data_dir_test='E:/kaggle/Solar/gefs_test/test/'):
    predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']
 
    train_suffix = '_latlon_subset_20080101_20121130.nc'    
    MesonetData,dates_test,stationdata = loadMesonetData(data_dir_test + 'sampleSubmission.csv')
#    randState = np.random.RandomState(123)
#    randDate = randState.randint(len(dates_test),size=50)
    dateTrain = np.empty((stationdata.shape[0],len(predictors)+1))
    predLen = len(predictors)
    finalArray=[]
    for ind in range(len(dates_test)):  
        date = dates_test[ind]  
        print date
        for i  in range(predLen):
            data = loadData(os.path.join(data_dir_test,predictors[i]+train_suffix))   
            grid = getDailyMeanSumGrid(data,date*100)
            outdata=buildSplines(data,grid,stationdata)
#            outdata = outdata.reshape(outdata.shape[0],1)         
            dateTrain[:,i+1] = outdata
        dateTrain[:,0] = date
        if ind == 0: 
            finalArray = deepcopy(dateTrain)
        else:
            finalArray = np.vstack((finalArray,dateTrain))
    f = open('spline_test.csv', 'w')
    np.savetxt(f, finalArray, delimiter=',',fmt='%7.5f')             
    f.close()
    data.close()

'''
	Description: Uses only the subsample of the whole training dataset
'''
def main(data_dir_train='E:/kaggle/Solar/gefs_train/train'):
    predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']
    train_suffix = '_latlon_subset_19940101_20071231.nc'
    dataPath = "E:/kaggle/Solar/gefs_test/test/"
    MesonetData,dates_test,stationdata = loadMesonetData(dataPath + 'sampleSubmission.csv')
    train_data = loadFromTrain(dataPath + 'train.csv')
    dates_train = loadTrainDates(train_data)
    randState = np.random.RandomState(123)
    randDate = randState.randint(len(dates_train),size=3000)
    dateTrain = np.empty((stationdata.shape[0],len(predictors)+2))
    predLen = len(predictors)
    finalArray=[]
    for ind in range(len(randDate)):  
        date = dates_train[randDate[ind]]  
        print date
        for i  in range(predLen):
            data = loadData(os.path.join(data_dir_train,predictors[i]+train_suffix))   
            grid = getDailyMeanSumGrid(data,date*100)
            outdata=buildSplines(data,grid,stationdata)
#            outdata = outdata.reshape(outdata.shape[0],1)         
            dateTrain[:,i+1] = outdata
        dateTrain[:,0] = date
        dateTrain[:,predLen+1] = dataForDate(train_data,date)
 #      tempArray = np.hstack((np.tile(dates_train[randDate[ind]], (stationdata.shape[0],1)),dateTrain))
        if ind == 0: 
            finalArray = deepcopy(dateTrain)
        else:
            finalArray = np.vstack((finalArray,dateTrain))
    f = open('spline_train3000.csv', 'w')
    np.savetxt(f, finalArray, delimiter=',',fmt='%7.5f')             
    f.close()
    data.close()
    

if __name__ == "__main__":
 #   main()
