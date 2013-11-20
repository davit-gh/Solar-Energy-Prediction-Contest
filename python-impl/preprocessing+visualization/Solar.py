import networkx as n
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from netCDF4 import Dataset
from mpl_toolkits.mplot3d import Axes3D 
from datetime import date
import math
import cProfile
import os
import pickle

'''
    Description: Class Solar contains methods for data preprocessing and visualization.
        Preprocessing comprises of finding nearest GEFS point to the given Mesonet station
        and then interpolating GEFS variables based on Mesonet station's lat/lon distance from GEFS point
        base. The given GEFS point may have one or more Mesonet stations.
'''
class Solar:
    
    _data_dir_train='E:/kaggle/Solar/gefs_train/train'
    _train_suffix = '_latlon_subset_19940101_20071231.nc'
    _dataPath = "E:/kaggle/Solar/gefs_test/test/"
    _predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

    def __init__(self):
        data = self.loadData('E:/kaggle/Solar/gefs_train/train/pres_msl_latlon_subset_19940101_20071231.nc')
        self.gefs_lat, self.gefs_lon = data.variables['lat'][3:7], data.variables['lon'][2:12] - 360
        self.mes_lat, self.mes_lon, self.mes_elev, self.stid = self.getData()
    
    def loadFromTrain(self, filename_train):
        train = np.genfromtxt(filename_train,delimiter=',',skiprows=1,dtype=float)
        return train
    
    def loadTrainDates(self, train):
        return np.array(train[:,0].T, dtype = int)

    def initGrid(self):
        featureArray = []
        dateArray = []
        train_data = self.loadFromTrain(self._dataPath + 'train.csv')
        train_dates = self.loadTrainDates(train_data)
        gfs = self.coordTest()
        grouped = []
         
        sgfs = sorted(gfs.items(), key = lambda x: x[1])
        groups = itertools.groupby(sgfs, key = lambda x: x[1])
         #     dist = []
        
        energiesAll = []
        
        for k, g in groups:
            grouped.append(list(g))
        grid = []
        for date in train_dates:
            
            for predictor in self._predictors:
                data = self.loadData(os.path.join(self._data_dir_train,predictor+self._train_suffix))  
                grid = self.getDailyMeanSumGrid(data,date * 100)[3:7,2:12].T
                energies = []
                for group in grouped:
                    subgroupEnergy = [self.interpolate(subgroup,grid) for subgroup in group]
                    energies.append(subgroupEnergy[0][0])
 #               with open('result' date,energies, len(energies)
                energiesAll.append(energies)
        with open('final.pickle','wb') as p: 
            pickle.dump(energiesAll, p)
        return 'Over'
 
    def compileTrainingData(self, fn = "E:/kaggle/Solar/gefs_test/test/train.csv"):
        data = pd.read_csv(fn, parse_dates = True, index_col=[0])
    #    start = date(1994,01,01); end = date(2007,12,31)
    #    rng = pd.date_range(start,end,freq=pd.DateOffset(years=1))
        grouped = data.groupby(lambda x: x.month)
        grouped_mean = grouped.mean()
        return grouped_mean
    def getData(self, fname = "E:/kaggle/Solar/gefs_train/train/station_info.csv"):
        dat = np.genfromtxt(fname,delimiter=',', dtype=[('stid','S4'), ('nlat',float), ('nlon',float), ('elev',float)],skiprows = 1)
        return dat['nlat'], dat['nlon'], dat['elev'], dat['stid']
    
    def plotMesonetwithAnnotation(self):
        plt.subplots_adjust(bottom = 0.1)
        
        plt.scatter(self.mes_lon,self.mes_lat,c = self.mes_elev, cmap=plt.get_cmap('rainbow'))
        for x, y in zip(self.mes_lat, self.mes_lon):
            plt.annotate('(%.1f,%.1f)'%(x,y), xy = (y, x), xytext=(-5,5), textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()
        
    def plotMesonetData(self):
        G = n.Graph()
        pos = {}
        labels = {}
        labels_pos = {}
        for i,j in enumerate(self.stid):
            G.add_node(i, name = j)
            pos[i] = (self.mes_lon[i], self.mes_lat[i])
            labels[i] = i#'%.1f, %.1f'%(lon[i], lat[i])
            labels_pos[i] = (self.mes_lon[i] - 0.1, self.mes_lat[i] - 0.1)
        mesonetPlot = n.draw_networkx_nodes(G, pos, cmap = plt.get_cmap('bone_r'), node_color = self.mes_elev, node_size=30)
        mesonetLabels = n.draw_networkx_labels(G,labels_pos,labels, font_color = 'm')
        return mesonetPlot, mesonetLabels
       # n.draw_networkx_labels(G, pos, alpha=0.7)
    
    def getGEFSLatLonElev(self):
        elev = []
        lon = []
        lat = []
        gefs = self.getGEFSdata()
        lat_length = len(gefs.dimensions['lat'])
        for i in range(3,lat_length-2):
    #        gefs_pos.append(zip(gefs.variables['longitude'][i][3:12]-360,gefs.variables['latitude'][i][3:12]))
            lon.extend(gefs.variables['longitude'][i][2:12] - 360)
            lat.extend(gefs.variables['latitude'][i][2:12])
            elev.extend(gefs.variables['elevation_control'][i][2:12])
        return lat, lon, elev

'''
	The following 2 functions are borrowd from benchmark code
'''
    def getDailyMeanSumGrid(self,data,date):
        dateIdx = np.where(data.variables['intTime'][:] == date)[0]
        fIdx = np.where(data.variables['fhour'][:] <= 24)[0]
        s = data.variables.values()[-1][dateIdx,:,fIdx,:,:]
        ret = s.sum(axis=2)
        retmean = ret.mean(axis=1)[0]
        return retmean
    def loadData(self, filename):
        data = Dataset(filename)
        return data


    def plotGEFSData(self):
        gefs_lat_cropped, gefs_lon_cropped, _ = self.getGEFSLatLonElev()
        node_pos = {}
        label_pos = {}
        labels = {}
#        data = loadData('E:/kaggle/Solar/gefs_train/train/pres_msl_latlon_subset_19940101_20071231.nc') 
#        grid = getDailyMeanSumGrid(self, data,1994010100)[2:8,2:13].flatten()
    #    gefs_pos = list(itertools.chain(*zip(lon,lat)))
        for key, val in enumerate(zip(gefs_lon_cropped,gefs_lat_cropped)):
            node_pos[key] = val
            label_pos[key] = (val[0], val[1] + 0.12) if key % 2 == 0 else (val[0], val[1] - 0.12)
           # labels[key] = elev[key] 
 #           labels[key] = grid[key]
     #   print [elem for x in gefs_pos for elem in x]   #clever
        Graph = n.path_graph(len(gefs_lat_cropped))
      #  plt.figure(1)
#        fig1 = n.draw_networkx_labels(Graph, label_pos, labels, font_color = 'b')
     #   plt.figure(2)
        fig2 = n.draw_networkx_nodes(Graph, node_pos, node_size = 60, node_shape='s', node_color = 'b')
        return  fig2
    
    def plot3d(self):
        energy = self.compileTrainingData()
        gefs_lat, gefs_lon, gefs_elev = self.getGEFSLatLonElev()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(self.gefs_lon, self.gefs_lat, gefs_elev, zdir='elev', s=40, c='b')#plot(xs = lon, ys = lat, zs = elev, zdir = 'elev', label = 'Looking good!')
        ax.scatter3D(self.mes_lon, self.mes_lat, self.mes_elev, zdir = 'mes_elev', s = 20, cmap =plt.get_cmap('RdYlGn_r'), c = 'r')
     #   fig.text(1, 1, plt.get_cmap('RdYlGn_r'), fontsize = 10, ha = 'right', va = 'bottom')
        plt.show()
    
    
    def getGEFSdata(self, fn = 'E:\kaggle\Solar\gefs_elevations.nc'):
        data = Dataset(fn)
        return data
    
    def calcDistance(self, lon1, lat1, lon2, lat2):
        R = 6371 #km
        d = math.acos(math.sin(lat1)*math.sin(lat2) + 
                      math.cos(lat1)*math.cos(lat2) *
                      math.cos(lon2-lon1)) * R;
        return d
    
    def coordTest(self):
#         gefs_lat_un = []; gefs_lon_un = []
        gefsForStid = {}
#         for k, g in itertools.groupby(self.gefs_lat):
#             #gefs_lat_un = np.append(gefs_lat_un, k)
#             gefs_lat_un.append(k)
#         for k, g in itertools.groupby(sorted(self.gefs_lon)):
#             gefs_lon_un.append(k)
        
        nearlat = []; nearlon = []
        mes_latLength = len(self.mes_lat)
        for i in range(mes_latLength):
            nearlat.append(np.where(np.abs(self.gefs_lat - self.mes_lat[i]) < 0.5))
            nearlon.append(np.where(np.abs(self.gefs_lon - self.mes_lon[i]) < 0.5))
        nearlat = [elem[0] for sub in nearlat for elem in sub]#list(itertools.chain(*nearlat)).map(lambda x: x[0])
        nearlon = [elem[0] for sub in nearlon for elem in sub]
        coords = zip(nearlon, nearlat)
 #       stidLength = len(self.stid)
        for i in range(mes_latLength):
            gefsForStid[self.stid[i]] = coords[i]
        return gefsForStid
    
    def computeDistance(self,group):
        idx = np.where(group[0] == self.stid)
        gefs_lon, gefs_lat = self.gefs_lon[group[1][0]], self.gefs_lat[group[1][1]]
        dist = self.calcDistance(self.mes_lon[idx], self.mes_lat[idx], gefs_lon, gefs_lat)
        return dist
    
    #here is the idea: first decide on which direction of GEFS point is: 2 checks 
    #1. if lat(gefs) - lat(mes) > 0 then below gefs else above gfs 2. lon(gefs) - lon(mes) > 0 then left else right
    # then calculate differences between 2 adjacent gefslats(diffgefslat) and 2 adjacent gefslons(diffgefslon) and depending on which size is the point: 
    #plus difflat*diffgefslat plus difflon*diffgefslon, plus minus, minus plus, minus minus
    def interpolate(self,group,grid):  
        mes_idx = np.where(group[0] == self.stid)
        gefs_lon_idx = group[1][0]
        gefs_lat_idx = group[1][1]
        gefs_lon, gefs_lat = self.gefs_lon[gefs_lon_idx], self.gefs_lat[gefs_lat_idx]
        difflat = gefs_lat - self.mes_lat[mes_idx]
        difflon = gefs_lon - self.mes_lon[mes_idx]
        currentGrid = grid[gefs_lon_idx][gefs_lat_idx]
        diffgefslat = grid[gefs_lon_idx][gefs_lat_idx - 1] - currentGrid
        diffgefslon = grid[gefs_lon_idx - 1][gefs_lat_idx] - currentGrid
        retVal = currentGrid + difflat * diffgefslat + difflon * diffgefslon
        return retVal#,  difflat, diffgefslat, difflon, diffgefslon
    