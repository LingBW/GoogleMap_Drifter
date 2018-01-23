# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:40:33 2017

@author: bling
"""

import sys
import math
#import pytz
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from global_currents_functions import get_drifter,get_fvcom,draw_basemap
from matplotlib import animation
#import io, json
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree,parse
#from ElementTree_pretty import prettify
from xml.dom import minidom
from mpl_toolkits.basemap import Basemap

st_run_time = datetime.now() # Caculate execution time with en_run_time

MODEL = 'global' #'global'GOM3
# Three options: backward, forward and both. 'both' only apply to Option 2 and 3.

######################################## Drifter ##################################

print 'Drifter parts.'

# Drifter loops
tree = parse('drift_X.xml')
#print type(tree)
root = tree.getroot()

dlastp = {} #[(lon,lat),,,]
for country in root.findall('marker'):   
    act = country.get('active')
    if act == '1' :
        #print act
        drifterid = country.get('label')
        lat = country.get('lat')
        lon = country.get('lng')
        #print drifterid,lat,lon
        dlastp[drifterid] = (float(lon),float(lat))

###################################### Plot ########################################
get_obj =  get_fvcom(MODEL)
get_obj.get_current_data()
# currets loop
currents = {}
for j in dlastp:
    cts = get_obj.current_track(dlastp[j][0],dlastp[j][1],0.3,7)
    currents[j] = cts
# Save xml file
top = Element('Currents')
#top.set('version', '1.0')
#comment = Comment('Generated for PyMOTW')
#top.append(comment)
for k in currents:
    #drifter = SubElement(top, k) #, num=str(i))
    for i in range(len(currents[k])):
        lines = SubElement(top, 'line',did=k) #, num=str(i))
        for j in currents[k][i]:
            points = SubElement(lines, 'point',{'lon':str(j[0]),'lat':str(j[1])})
            #lat = SubElement(child1, 'lat')
            #for j in i:
            #gk = np.vstack(i).T
            #points.text = str(cts[i])
            #lat.text = str(gk[1])
#top.toprettyxml()
tos = ElementTree(top)
tos.write('myxmlfile.xml')#'''
#reparsed = minidom.parseString(tos)
#print reparsed.toprettyxml() 
#f =  open("myxmlfile.xml", "wb")
#f.write('<?xml version="1.0" encoding="UTF-8"?>')
#f.write(tos)
#f.close()
