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
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
'''
    Description: Class Solar contains methods for data preprocessing and visualization.
        Preprocessing comprises of finding nearest GEFS point to the given Mesonet station
        and then interpolating GEFS variables based on Mesonet station's lat/lon distance from GEFS point
        base. The given GEFS point may have one or more Mesonet stations.
'''
class Solar:
    
    _data_dir_train='E:/kaggle/Solar/gefs_train/train'
    _train_suffix = '_latlon_subset_19940101_20071231.nc'
    _dataPath = "gefs/"
    _predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

    def __init__(self):
        self.data = self.loadData('gefs/train/pres_msl'+self._train_suffix)
        self.gefs_lat, self.gefs_lon = self.data.variables['lat'][3:7], self.data.variables['lon'][2:12] - 360
        self.mes_lat, self.mes_lon, self.mes_elev, self.stid = self.getData()
    
    def loadFromTrain(self, filename_train):
        train = np.genfromtxt(filename_train,delimiter=',',skiprows=1,dtype=float)
        return train
    
    def loadTrainDates(self, train):
        return np.array(train[:,0].T, dtype = int)

    
    def interpolated(self,gfs):
        interpltd = {}
        for gf in gfs:
            subgroupEnergy = self.interpolate(gf,gfs[gf],grid)
            interpolated[gf] = subgroupEnergy[0]
        return interpltd
            
    
 
    
    def compileTrainingData(self, fn = "gefs/train.csv"):
        data = pd.read_csv(fn, parse_dates = True, index_col=[0])
        return data
    def getData(self, fname = "gefs/station_info.csv"):
        dat = np.genfromtxt(fname,delimiter=',', dtype=[('stid','S4'), ('nlat',float), ('nlon',float), ('elev',float)],skiprows = 1)
        return dat['nlat'], dat['nlon'], dat['elev'], dat['stid']
    
    def plotMesonetwithAnnotation(self):
        plt.subplots_adjust(bottom = 0.1)
        
        plt.scatter(self.mes_lon,self.mes_lat,c = self.mes_elev, cmap=plt.get_cmap('rainbow'))
        for x, y in zip(self.mes_lat, self.mes_lon):
            plt.annotate('(%.1f,%.1f)'%(x,y), xy = (y, x), xytext=(-5,5), textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()
    
    def getGroups(self):
        gfs = self.coordTest()
        grouped = [] 
        sgfs = sorted(gfs.items(), key = lambda x: x[1])
        groups = itertools.groupby(sgfs, key = lambda x: x[1])
        d = {}
         
        for k, g in groups:
          grouped.append(list(g))
        for i,group in enumerate(grouped):
          for item in group:
            d[item[0]] = i
        return d, grouped
    
       # n.draw_networkx_labels(G, pos, alpha=0.7)
    
    def plotMesonetData(self, stations,energies):
        lst = [sub[0] for group in stations for sub in group]
        G = n.Graph()
        pos = {}
        labels = {}
        labels_pos = {}
        for j in lst:
            i = np.where(self.stid == j)
            G.add_node(j)
            pos[j] = (self.mes_lon[i], self.mes_lat[i])
            labels[j] = '%.1f'%energies[j]#'%.1f, %.1f'%(lon[i], lat[i])
            labels_pos[j] = (self.mes_lon[i] - 0.1, self.mes_lat[i] - 0.1)
#       
        
#        
#         cNorm  = colors.Normalize(vmin=0, vmax=values[-10])
#         scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet'))
#         colorList = []
#         for i in values:
#           colorVal = scalarMap.to_rgba(values[i])
#           colorList.append(colorVal)
        
        values = range(len(stations))
        d={}
        for i in values:
            for it in stations[i]:
                d[it[0]] = i
        col = [d.get(v) for v in G.nodes()]
        
        mesonetPlot = n.draw_networkx_nodes(G, pos, node_color = col, node_size=50)
        mesonetLabels = n.draw_networkx_labels(G,labels_pos,labels, font_color = 'm')
        return mesonetPlot, mesonetLabels

    def getGEFSLatLonElev(self):
        elev = []
        lon = []
        lat = []
        gefs = self.getGEFSdata()
        lat_length = len(gefs.dimensions['lat'])
        for i in range(3,lat_length-2):
    #        gefs_pos.append(zip(gefs.variables['longitude'][i][3:12]-360,gefs.variables['latitude'][i][3:12]))
            lon.extend(gefs.variables['longitude'][i][2:13] - 360)
            lat.extend(gefs.variables['latitude'][i][2:13])
            elev.extend(gefs.variables['elevation_control'][i][2:13])
        return lat, lon, elev
    
    def get5grids(self,data,date,ens):
 
        date_idx = np.where(data.variables['intTime'][:] == date)[0][0]
        hours = data.variables['fhour'][:]
        target = data.variables.values()[-1]
        
        return [target[date_idx,0,h,:,:] for h in range(5)]# 5 == range(len(hours))

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


    def plotGEFSData(self,grid):
        gefs_lat_cropped, gefs_lon_cropped, _ = self.getGEFSLatLonElev()
        node_pos = {}
        label_pos = {}
        labels = {}
#        data = loadData('E:/kaggle/Solar/gefs_train/train/pres_msl_latlon_subset_19940101_20071231.nc') 
        #grid = self.getDailyMeanSumGrid(self.data,1994010100)[2:8,2:13].flatten()
   
    #    gefs_pos = list(itertools.chain(*zip(lon,lat)))
        lonlatzip = zip(gefs_lon_cropped,gefs_lat_cropped)
       # print lonlatzip
        for key, val in enumerate(lonlatzip):
            node_pos[key] = val
            label_pos[key] = (val[0], val[1] + 0.12) if key % 2 == 0 else (val[0], val[1] - 0.12)
           # labels[key] = elev[key] 
            labels[key] = grid[key]
     #   print [elem for x in gefs_pos for elem in x]   #clever
        Graph = n.path_graph(len(gefs_lat_cropped))
     #   plt.figure(1)
        fig1 = n.draw_networkx_labels(Graph, label_pos, labels, font_color = 'b')
     #   plt.figure(2)
        fig2 = n.draw_networkx_nodes(Graph, node_pos, node_size = 60, node_shape='s', node_color = 'k')
        return  fig1, fig2
  
        
    def plot3d(self):
        energy = self.compileTrainingData()
        gefs_lat, gefs_lon, gefs_elev = self.getGEFSLatLonElev()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(self.gefs_lon, self.gefs_lat, gefs_elev, zdir='elev', s=40, c='b')#plot(xs = lon, ys = lat, zs = elev, zdir = 'elev', label = 'Looking good!')
        ax.scatter3D(self.mes_lon, self.mes_lat, self.mes_elev, zdir = 'mes_elev', s = 20, cmap =plt.get_cmap('RdYlGn_r'), c = 'r')
     #   fig.text(1, 1, plt.get_cmap('RdYlGn_r'), fontsize = 10, ha = 'right', va = 'bottom')
        plt.show()
    
    
    def getGEFSdata(self, fn = 'gefs/gefs_elevations.nc'):
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
    def setFromCommonGEFS(self,sg,grid):
        gefs_lon_idx = sg[1][0]
        gefs_lat_idx = sg[1][1]
        gefs_lon, gefs_lat = self.gefs_lon[gefs_lon_idx], self.gefs_lat[gefs_lat_idx]
        currentGrid = grid[gefs_lon_idx][gefs_lat_idx]
        return currentGrid
    
    def interpolate(self,group0,group1,grid):  
        mes_idx = np.where(group0 == self.stid)
        gefs_lon_idx = group1[0]
        gefs_lat_idx = group1[1]
        gefs_lon, gefs_lat = self.gefs_lon[gefs_lon_idx], self.gefs_lat[gefs_lat_idx]
        difflat = gefs_lat - self.mes_lat[mes_idx]
        difflon = gefs_lon - self.mes_lon[mes_idx]
        currentGrid = grid[gefs_lon_idx][gefs_lat_idx]
        diffgefslat = grid[gefs_lon_idx][gefs_lat_idx - 1] - currentGrid
        diffgefslon = grid[gefs_lon_idx - 1][gefs_lat_idx] - currentGrid
        retVal = currentGrid + difflat * diffgefslat + difflon * diffgefslon
        return retVal#,  difflat, diffgefslat, difflon, diffgefslon
    
    def init(self,datass,date,sfgs):
        grids15x5 = []
        for data in datass:
            grids = [grid[2:8,2:13].T for grid in self.get5grids(data,int(date.strftime('%Y%m%d'))*100,0)]
            grids15x5.append(grids)

        
        return [self.initGrid(grid,sgfs) for grids5 in grids15x5 for grid in grids5]
    
    
    def initGrid(self,grid, sgfs):    
        clustered = {}
        
        groups = itertools.groupby(sgfs, key = lambda x: x[1])             
        for k, g in groups:
            subgroup = list(g)
          
            subgroupEnergy = self.setFromCommonGEFS(subgroup[0],grid)   #     data = self.loadData(os.path.join(self._data_dir_train,_predictor+self._train_suffix))  
            for subsub in subgroup:    
                clustered[subsub[0]] = subgroupEnergy
                              
        return  clustered