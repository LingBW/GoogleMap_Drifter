# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:43:50 2017

@author: bling
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:33:53 2016

@author: Bingwei Ling
"""

import sys
import netCDF4
#import calendar
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from dateutil.parser import parse
#import pytz
from matplotlib.path import Path
import math
from mpl_toolkits.basemap import Basemap
import colorsys
from sympy import *
from sympy.geometry import *
from fractions import Fraction


def get_nc_data(url, *args):
    '''
    get specific dataset from url

    *args: dataset name, composed by strings
    ----------------------------------------
    example:
        url = 'http://www.nefsc.noaa.gov/drifter/drift_tcs_2013_1.dat'
        data = get_url_data(url, 'u', 'v')
    '''
    nc = netCDF4.Dataset(url)
    data = {}
    for arg in args:
        try:
            data[arg] = nc.variables[arg]
        except (IndexError, NameError, KeyError):
            print 'Dataset {0} is not found'.format(arg)
    return data
    
class get_fvcom():
    
    def __init__(self, mod):
        self.modelname = mod
    def points_square(self,point, hside_length):
        '''point = (lat,lon); length: units is decimal degrees.
           return a squre points(lats,lons) on center point,without center point
           hside_length is radius.'''
        ps = []
        (lat,lon) = point; 
        length =float(hside_length)
        #lats=[lat]; lons=[lon]
        #lats=[]; lons=[]
        bbox = [lon-length, lon+length, lat-length, lat+length]
        bbox = np.array(bbox)
        points = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]])
        #print points
        pointt = points.T
        for i in pointt:
            ps.append((i[1],i[0]))
        ps.append((pointt[0][1],pointt[0][0]))# add first point one more time for Path.
        #lats.extend(points[1]); lons.extend(points[0])
        #bps = np.vstack((lon,lat)).T
        #return lats,lons
        return ps
    
    def get_current_data(self):
        '''
        "get_data" not only returns boundary points but defines global attributes to the object
        '''
        if self.modelname == "global":
            turl = '''http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_GLOBAL_FORECAST.nc'''
        if self.modelname == "GOM3":
            turl = '''http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'''
               
        try:
            self.tdata = netCDF4.Dataset(turl).variables
            #MTime = tdata['time'] 
            #MTime = self.tdata['Times']
        except:
            print '"massbay" database is unavailable!'
            raise Exception()
        
        #Times = netCDF4.num2date(MTime[:],MTime.units)
       
        #self.data = get_nc_data(url,'lat','lon','latc','lonc','siglay','h','nbe','u','v','zeta')#,'nv'
        loncs = self.tdata['lonc'][:]; self.latc = self.tdata['latc'][:]  #quantity:global,689132
        #print type(loncs), loncs.shape
        for i in range(len(loncs)):
            if loncs[i] > 180:
                loncs[i] = loncs[i]-360
        self.lonc = loncs
        #self.lons, self.lats = self.data['lon'][:], self.data['lat'][:]
        #self.h = self.data['h'][:]; self.siglay = self.data['siglay'][:]; #nv = self.data['nv'][:]
        #self.u = self.tdata['u'][:,0,:][:]; self.v = self.tdata['v'][:,0,:][:]#; self.zeta = self.data['zeta']
        self.u = self.tdata['u'][3,0,:]; self.v = self.tdata['v'][3,0,:]
        self.gmap=Basemap(projection='cyl',resolution='h')
        #return psqus #self.b_points,,nv lons,lats,lonc,latc,,h,siglay
    
    def extend_units(self,point, unit, num):
        '''point = (lat,lon); length: units is decimal degrees.
           return a squre points[[minlon,maxlon,minlat,maxlat],,,]'''
        (lon,lat) = point; 
        lats=[]; lons=[]; sqs=[]
        #unit = unit
        leh = unit*num
        hu = unit/2.0
        #ps = np.mgrid[lat-leh:lat+leh+1:2j, lon-leh:lon+leh+1:2j]; print ps
        #ps = ps*unit
        lts = np.arange(lat-leh,lat+leh+unit,unit); lns = np.arange(lon-leh,lon+leh+unit,unit)
        llp = np.meshgrid(lns,lts)#; print llp
        lats.extend(llp[1].flatten()); lons.extend(llp[0].flatten())
        pp = np.vstack((lons,lats)).T
        for i,j in pp:
            sqs.append([i-hu,i+hu,j-hu,j+hu])
        return sqs
        
    def current_track(self,lon,lat,unit,num): #,point,leh):
        
        cts = []        
        le = unit*num
        psqus = self.points_square((lon,lat),le) # Got four point of rectangle with center point (lon,lat)        
        codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
        sp = Path(psqus,codes)
        self.gmap=Basemap(projection='cyl',llcrnrlat=lat-le, llcrnrlon=lon-le,urcrnrlat=lat+le, urcrnrlon=lon+le,resolution='h')
        
        pit = (lon,lat)
        squares = self.extend_units(pit, unit, num)
        ln = len(squares)
        for i in range(ln):
            #ct = []; gk = []
            print '%d of %d' % (i+1,ln)
            #print self.lonc[i][0],self.lonc[i][0]
            getk = self.get_current(squares[i],unit,sp)
            if len(getk)<3 :
                continue
            print len(getk)
            cts.append(getk)
            #ct.append(g)
            #cts["%s"%str(i)]=g
        return cts# [[(),()..],[(),()..]..]
        
    def get_current(self,sip,ut,sp): #,b_index,nvdepth,,bcon 
        '''
        Get forecast points start at lon,lat
        '''
        
        inds = np.argwhere((self.lonc >= sip[0]) & (self.lonc <= sip[1]) & (self.latc >= sip[2]) & (self.latc <= sip[3]))
        #print 'inds',inds
        if len(inds) == 0:
            return []
        lon = np.mean(self.lonc[inds]); lat = np.mean(self.latc[inds])
        if self.gmap.is_land(lat,lon):
            return []
        modpts = [(lon,lat)]#;st = []
        #self.lonk,self.latk,sp = self.shrink_data_circle(lon,lat,self.lonc,self.latc,30)
        #modpts = dict(lon=[lon], lat=[lat], time=[], spd=[]) #model forecast points, layer=[]
            
        #t = abs(self.days) 
    
        for i in xrange(50): 
            
            u_t1 = np.mean(self.u[inds])
            v_t1 = np.mean(self.v[inds])
        
            if self.modelname == "GOM3":
                dx = 60*60*u_t1; dy = 60*60*v_t1
            if self.modelname == "global":
                dx = 12*60*60*u_t1; dy = 12*60*60*v_t1
            #pspeed = math.sqrt(u_t1**2+v_t1**2)
            #modpts['spd'].append(pspeed)
                     
            lon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            lat = lat + dy/111111 #'''   
            fpoint = (lon,lat); modpts.append(fpoint)
            
            # Condition 1
            '''if sd < 0.02:
                print 'Low u,v',u_t1,v_t1
                return modpts#'''
            # Condition 2   
            if not sp.contains_point(fpoint):                
                return modpts#,tdexs''' 
            # Condition 3
            if self.gmap.is_land(lat,lon):
                return modpts
                
            inds = np.argwhere((self.lonc >= lon-ut) & (self.lonc <= lon+ut) & (self.latc >= lat-ut) & (self.latc <= lat+ut))
            if len(inds) == 0:
                return modpts
            #modpts.append((lon,lat)) #; modpts['layer'].append(layer); 
        return modpts
        
def draw_basemap(ax, points, interval_lon=0.5, interval_lat=0.5):
    '''
    draw the basemap?
    '''
    '''
    lons = points['lons']
    lats = points['lats']
    #size = max((max(lons)-min(lons)),(max(lats)-min(lats)))/2
    size = 0
    map_lon = [min(lons)-size,max(lons)+size]
    map_lat = [min(lats)-size,max(lats)+size]#'''
    
    map_lon = [points[0],points[1]]
    map_lat = [points[2],points[3]]
    #ax = fig.sca(ax)
    dmap = Basemap(projection='cyl',
                   llcrnrlat=map_lat[0], llcrnrlon=map_lon[0],
                   urcrnrlat=map_lat[1], urcrnrlon=map_lon[1],
                   resolution='h',ax=ax)# resolution: c,l,i,h,f.
    dmap.drawparallels(np.arange(int(map_lat[0])-1,
                                 int(map_lat[1])+1,interval_lat),
                       labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(map_lon[0])-1,
                                 int(map_lon[1])+1,interval_lon),
                       labels=[0,0,0,1])
    #dmap.drawcoastlines()
    #dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    #dmap.etopo()
    
class get_drifter():

    def __init__(self, drifter_id, filename=None):
        self.drifter_id = drifter_id
        self.filename = filename
        
    def getrawdrift(self, did,filename):
        
        '''
        routine to get raw drifter data from ascii files posted on the web
        '''
        url='http://nefsc.noaa.gov/drifter/'+filename
        df=pd.read_csv(url,header=None, delimiter="\s+")
        # make a datetime
        dtime=[]
        index = np.where(df[0]==int(did))[0]
        newData = df.ix[index]
        for k in newData[0].index:
            dt1=datetime(2017, newData[2][k],newData[3][k],newData[4][k],newData[5][k])
            dtime.append(dt1)
        #print dtime
        return newData[8],newData[7],dtime,newData[9] # lat,lon,time,

    def getdrift(self,did):
        
        """
        routine to get drifter data from archive based on drifter id (did)
        -assumes "import pandas as pd" has been issued above
        -get remotely-stored drifter data via ERDDAP
        -input: deployment id ("did") number where "did" is a string
        -output: time(datetime), lat (decimal degrees), lon (decimal degrees), depth (meters)
        
        note: there is another function below called "data_extracted" that does a similar thing returning a dictionary
        
        Jim Manning June 2014
        """
        url = 'http://comet.nefsc.noaa.gov:8080/erddap/tabledap/drifters.csv?time,latitude,longitude,depth&id="'+did+'"&orderBy("time")'
        df=pd.read_csv(url,skiprows=[1]) #returns a dataframe with all that requested
        #print df    
        # generate this datetime 
        for k in range(len(df)):
           df.time[k]=parse(df.time[k]) # note this "parse" routine magically converts ERDDAP time to Python datetime
        return df.latitude.values,df.longitude.values,df.time.values,df.depth.values 
        
    def get_track(self):
        '''
        return drifter nodes
        if starttime is given, return nodes started from starttime
        if both starttime and days are given, return nodes of the specific time period
        '''
        if self.filename:
            temp = self.getrawdrift(self.drifter_id,self.filename)
        else:
            temp = self.getdrift(self.drifter_id)
        nodes = {}
        nodes['lon'] = np.array(temp[1])
        nodes['lat'] = np.array(temp[0])
        
        return (nodes['lat'][-1],nodes['lon'][-1])
        
    def __cmptime(self, time, times):
        '''
        return indies of specific or nearest time in times.
        '''
        tdelta = []
        #print len(times)
        for t in times:
            tdelta.append(abs((time-t).total_seconds()))
            
        index = tdelta.index(min(tdelta))
        
        return index

