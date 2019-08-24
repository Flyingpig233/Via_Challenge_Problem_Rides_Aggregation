
# coding: utf-8

# ### Implement the algorithm

# In[9]:


import numpy as np
import pandas as pd
import math
import time
import datetime


# In[10]:


data = pd.read_csv('yellow_tripdata_2016-06.csv')


# In[11]:


df = data.copy()


# In[12]:


# filter records with pickup time before dropoff time
df = df[pd.to_datetime(df['tpep_dropoff_datetime']) > pd.to_datetime(df['tpep_pickup_datetime'])]
time_diff = pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])


# In[13]:


# trip_duration in minutes
df['trip_duration'] = [round(t.total_seconds()/60,0) for t in time_diff]

# deal with records the trip duration is 0 min
speed_calculate = df[df['trip_duration']!=0][['trip_duration','trip_distance']]

# calculate average speed
speed = sum(speed_calculate['trip_distance']/speed_calculate['trip_duration'])/speed_calculate.shape[0]
print('Average driver speed is: ', speed,'/min')


# In[16]:


# preparedata for algorithm
df2 = df[df['trip_duration']!=0][['tpep_pickup_datetime', 'tpep_dropoff_datetime',
       'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
df2 = df2.sort_values(by=['tpep_pickup_datetime'])

# 06/06/2016 - 06/12/2016  the first full week, Monday to Sunday
df3 = df2[(df2['tpep_pickup_datetime']>='2016-06-06')& (df2['tpep_pickup_datetime']<'2016-06-13')]
df3.shape


# In[20]:


# use longitude latitude to calculate distance (in miles)
EARTH_REDIUS = 3958.8
def rad(d):
    return d * math.pi / 180.0

def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s

# not equal to the meter records on vehicles
def getManhattanDistance(lat1, lng1, lat2, lng2):
    return getDistance(lat1,lng1,lat2,lng1)+getDistance(lat2,lng1,lat2,lng2)


# In[21]:


# use binary search to find the last passenger who meet the time requirement
def findborder(b,l,r):
    mid=(l+r)//2
    while(r-l>1):
        if b>=pktime[mid]:
            l=mid
        else:
            r=mid
        mid=(l+r)//2
    return mid


# In[22]:


# whether (x,y) is in the rectangular with diagonal lining by (x1,y1),(x2,y2)
def notthearea(x,y,x1,y1,x2,y2):
    if (x>x1) & (x>x2):
        return 1
    if (x<x1) & (x<x2):
        return 1
    if (y<y1) & (y<y2):
        return 1
    if (y>y1) & (y>y2):
        return 1
    return 0


# In[23]:


# main function: find qualified share-ride passenger for i th passenger ordered by pick-up time

def find_share_rides_passenger(i):
    
    # ride information of i th passenger
    pickup_time, dropoff_time, passenger_num, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat = da.loc[i][:7]
    dropoff_time_t=pd.to_datetime(dropoff_time)
    
    # find potential share-ride passenger who meets time requirement
    bor=findborder(dropoff_time_t,i+1,n)
    arr=da[i+1:bor]
    
    # find the first one who meets route requirement
    for data in arr.itertuples():
        j=data[0]
        
        # exceed the maximum passenger capacity or already get on other vehicles, move to next
        # finished_passenger_tag is a global variable, indicae whether a massenger already gets on vehicle or not
        if ((data.passenger_count+passenger_num>4) | (finished_passenger_tag[j]==1)):
            continue
            
        # otherwise, check route requiement
        pickup2_lon, pickup2_lat, dropoff2_lon, dropoff2_lat=data[4:8]
        
        # pick-up spot not qualified, move to next
        if notthearea(pickup2_lon, pickup2_lat,pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
            continue
            
        # decide who to drop off first 
        drop2first=not(notthearea(dropoff2_lon, dropoff2_lat,pickup2_lon, pickup2_lat, dropoff_lon, dropoff_lat))
        drop1first=not(notthearea(dropoff_lon, dropoff_lat,pickup2_lon, pickup2_lat,dropoff2_lon, dropoff2_lat))
        
        # calculate corresponding distance
        if (drop2first) | (drop1first):
            distance1 = passenger_dis[i] + data.passenger_dis
            if drop2first:
                distance2 =  passenger_dis[i]
                finished_passenger_tag[j]=1
                return [i,j,distance1, distance2,[(pickup_lon, pickup_lat),(pickup2_lon, pickup2_lat),(dropoff2_lon, dropoff2_lat),(dropoff_lon, dropoff_lat)]]
            else:
                distance2 =  passenger_dis[i] + getManhattanDistance(dropoff_lon, dropoff_lat,dropoff2_lon, dropoff2_lat)
                finished_passenger_tag[j]=1
                return [i,j,distance1, distance2,[(pickup_lon, pickup_lat),(pickup2_lon, pickup2_lat),(dropoff_lon, dropoff_lat),(dropoff2_lon, dropoff2_lat)]]
    
    # if no qualified share-ride passengers, return -1, otherwise return the index of share-ride passenger
    return [i,-1, passenger_dis[i], passenger_dis[i],[(pickup_lon, pickup_lat),(dropoff_lon, dropoff_lat)]]


# In[24]:


# df3 is ordered by pickup time
da = df3.reset_index(drop=True)
n = da.shape[0]
pktime=pd.to_datetime(da['tpep_pickup_datetime'])

# calculate distance of each passengers' ride 
passenger_dis=[]
for i in da.itertuples():
    pickup_lon, pickup_lat, dropoff_lon, dropoff_lat = [i[4],i[5],i[6],i[7]]
    passenger_dis.append(getManhattanDistance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon))
da['passenger_dis']=passenger_dis

# indicator for finished passengers
finished_passenger_tag=np.zeros(n)


# In[25]:


# save and load part of the results during running of the algorithm
import pickle
def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r


# In[18]:



l = []
a=time.time()

for data in da.itertuples():
    i=data[0]
    if i%100000==0:
        tt=time.time()-a
        print(i,' 100000times:',tt,' est time: ', tt*(n-i)/6000000,'mins')
        a=time.time()
        save_variable(l,str(i)+'.txt')
        save_variable(finished_passenger_tag,'finished_passenger_tag'+str(i))
    if finished_passenger_tag[i]: continue
    finished_passenger_tag[i]=1
    tmp=find_share_rides_passenger(i)  #[-1, passenger_a, passenger_a] or [j, distance1, distance2]
    l.append(tmp)
n_vehicle = len(l)

# total miles when not aggregate vehicles
d_1 = 0
# total miles when aggregate vehicles
d_2 = 0

# calculate efficiency
for i in range(n_vehicle):
    d_1 = l[i][2] + d_1
    d_2 = l[i][3] + d_2
print([d_1, d_2,d_2/d_1])


# In[19]:


save_variable(l,'finished.txt')


# In[20]:


save_variable(pktime,'time.txt')


# In[8]:


pktime=load_variable('time.txt')


# ### Measure how efficiency varies according to time

# In[9]:


# find slices of data during different time period
last='2016-06-06 00:00:00'
lst=[]
for day in ['06','07','08','09','10','11','12']:
    for hour in['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']:
        time='2016-06-%s %s:00:00'%(day,hour)
        if time=='2016-06-06 00:00:00':
            continue
        a=(pktime>=last) & (pktime<time)
        lst.append([time,min(a[a].index),max(a[a].index)])
        last=time
lst.append(['2016-06-13 00:00:00',max(a[a].index)+1,len(a)+1])


# In[11]:


# measure total distance
k=0
oldroute=0
newroute=0
lll=[]
tenk=l
for ll in tenk:
    if (ll[0]<=lst[k][2]) & (ll[0]>=lst[k][1]):
        oldroute=oldroute+ll[2]
        newroute=newroute+ll[3]
    else:
        lll.append([lst[k][0],oldroute,newroute,newroute/oldroute])
        oldroute=ll[2]
        newroute=ll[3]
        k=k+1


# In[19]:


# measure total cars
k=0
totalcar=0
totalcarnow=0
lll=[]
tenk=l
for ll in tenk:
    if (ll[0]<=lst[k][2]) & (ll[0]>=lst[k][1]):
        totalcar=totalcar+1
        totalcarnow=totalcarnow+1
        if (ll[2]<ll[3]):
            totalcar=totalcar+1
    else:
        lll.append([lst[k][0],totalcar,totalcarnow,totalcarnow/totalcar])
        totalcarnow=1
        totalcar=1
        if (ll[2]<ll[3]):
            totalcar=totalcar+1
        k=k+1


# In[20]:


for mmm in lll:
    print(mmm[0],';',mmm[1],';',mmm[2],';',mmm[3])


# In[35]:


l=load_variable('finished.txt')


# ### Measure how efficiency varies according to location

# In[36]:


import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.speedups
from shapely.geometry import Point, Polygon
geo_data=gpd.read_file('geo_export_ad8f6470-4f30-4869-a3e2-27a889288479.shp')
bronx=geo_data[geo_data['boro_name']=='Bronx']['geometry'][0]
staten_island = geo_data[geo_data['boro_name']=='Staten Island']['geometry'][1]
queens = geo_data[geo_data['boro_name']=='Queens']['geometry'][2]
manhattan = geo_data[geo_data['boro_name']=='Manhattan']['geometry'][3]
brooklyn = geo_data[geo_data['boro_name']=='Brooklyn']['geometry'][4]


# In[50]:


# decide where the passenger comes from 
def find_location(x):
    X = Point(x)
    if X.within(manhattan): return 'Manhattan'
    if X.within(queens): return 'Queens'
    if X.within(brooklyn): return 'Brooklyn'
    if X.within(bronx): return 'Bronx'
    if X.within(staten_island): return 'Staten Island'
    else: return 'Other'


# In[ ]:


# calculate efficiency by areas
Bronx_noshare=0
Bronx_share=0
SI_noshare=0
SI_share=0
Queens_noshare=0
Queens_share=0
Manhattan_noshare=0
Manhattan_share=0
brooklyn_noshare=0
brooklyn_share=0
le=len(l)
for ele in l:
    index=ele[0]
    if index%10000<5:
        print(index)
    ride=da.loc[index]
    lon=ride.pickup_longitude
    lat=ride.pickup_latitude
    
    # find slices of passengers' data from different area
    area=find_location((lon,lat))
    if area=='Bronx':
        Bronx_noshare=Bronx_noshare+ele[2]
        Bronx_share=Bronx_share+ele[3]
    elif area=='Staten Island':
        SI_noshare=SI_noshare+ele[2]
        SI_share=SI_share+ele[3]
    elif area=='Queens':
        Queens_noshare=Queens_noshare+ele[2]
        Queens_share=Queens_share+ele[3]
    elif area=='Manhattan':
        Manhattan_noshare=Manhattan_noshare+ele[2]
        Manhattan_share=Manhattan_share+ele[3]
    elif area=='Brooklyn':
        brooklyn_noshare=brooklyn_noshare+ele[2]
        brooklyn_share=brooklyn_share+ele[3]


# In[ ]:


#pick up area efficiency
print('Bronx eff:',Bronx_share/Bronx_noshare,Bronx_share,Bronx_noshare)
print('Staten Island eff:',SI_share/SI_noshare,SI_share,SI_noshare)
print('Queens eff:',Queens_share/Queens_noshare,Queens_share,Queens_noshare)
print('Manhattan eff:',Manhattan_share/Manhattan_noshare,Manhattan_share,Manhattan_noshare)
print('Brooklyn eff:',brooklyn_share/brooklyn_noshare,brooklyn_share,brooklyn_noshare)
#Bronx eff: 0.8785164473328837
#Staten Island eff: 0.9710477606797943
#Queens eff: 0.8649045985198403
#Manhattan eff: 0.8867857406461953
#Brooklyn eff: 0.901239790502349

