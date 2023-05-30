import sys
import numpy as np
import numpy.ma as ma
import scipy as sp
#from   zCal import zCam
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import datetime
from ampang import AmpPhsDiagram

###obs####
iyr=10
uao=np.zeros([iyr])
uto=np.zeros([iyr])
vao=np.zeros([iyr])
vto=np.zeros([iyr])
for i in range(10):
   uuu=np.loadtxt('./chuv/utid'+str(i)+'.txt')
   vvv=np.loadtxt('./chuv/vtid'+str(i)+'.txt')
   uao[i]=uuu[2,0]
   uto[i]=uuu[2,1]
   vao[i]=vvv[2,0]
   vto[i]=vvv[2,1]
###cal####
jyr=20
uac=np.zeros([jyr])
utc=np.zeros([jyr])
vac=np.zeros([jyr])
vtc=np.zeros([jyr])
for i in range(20):
   uuu=np.loadtxt('./caluv/u'+str(i)+'020.txt')
   vvv=np.loadtxt('./caluv/v'+str(i)+'020.txt')
   uac[i]=uuu[2,0]
   utc[i]=uuu[2,1]
   vac[i]=vvv[2,0]
   vtc[i]=vvv[2,1]
###4-20####
aa=1+0j
bb=0+1j
pih=np.pi/180.

ctov11 =vao*np.exp(vto*pih*bb)
ctou11 =uao*np.exp(uto*pih*bb)
ctoval = np.average(ctov11[0:9])
ctoual = np.average(ctou11[0:9])
ctcv11 =vac*np.exp(vtc*pih*bb)*100.
ctcu11 =uac*np.exp(utc*pih*bb)*100.
ctcval = np.average(ctcv11[0:14])
ctcual = np.average(ctcu11[0:14])

wpo11= (ctou11.conjugate()+bb*ctov11.conjugate())/2.
wmo11=((ctou11.conjugate()-bb*ctov11.conjugate())/2.).conjugate()
wpoal= (ctoual.conjugate()+bb*ctoval.conjugate())/2.
wmoal=((ctoual.conjugate()-bb*ctoval.conjugate())/2.).conjugate()
wpc11= (ctcu11.conjugate()+bb*ctcv11.conjugate())/2.
wmc11=((ctcu11.conjugate()-bb*ctcv11.conjugate())/2.).conjugate()
wpcal= (ctcual.conjugate()+bb*ctcval.conjugate())/2.
wmcal=((ctcual.conjugate()-bb*ctcval.conjugate())/2.).conjugate()

to11=(np.arctan2(wpo11.imag,wpo11.real)+np.arctan2(wmo11.imag,wmo11.real))/2.
toal=(np.arctan2(wpoal.imag,wpoal.real)+np.arctan2(wmoal.imag,wmoal.real))/2.
tc11=(np.arctan2(wpc11.imag,wpc11.real)+np.arctan2(wmc11.imag,wmc11.real))/2.
tcal=(np.arctan2(wpcal.imag,wpcal.real)+np.arctan2(wmcal.imag,wmcal.real))/2.

to11=to11+np.pi
toal=toal+np.pi
tc11=tc11+np.pi    
tcal=tcal+np.pi

so11=to11-np.arctan2(wmo11.imag,wmo11.real)
soal=toal-np.arctan2(wmoal.imag,wmoal.real)
sc11=tc11-np.arctan2(wmc11.imag,wmc11.real)
scal=tcal-np.arctan2(wmcal.imag,wmcal.real)

ro11=np.cos(-so11)*aa+np.sin(-so11)*bb
roal=np.cos(-soal)*aa+np.sin(-soal)*bb
rc11=np.cos(-sc11)*aa+np.sin(-sc11)*bb
rcal=np.cos(-scal)*aa+np.sin(-scal)*bb


wao11=np.cos(to11)*ctou11 + np.sin(to11)*ctov11
waoal=np.cos(toal)*ctoual + np.sin(toal)*ctoval
wac11=np.cos(tc11)*ctcu11 + np.sin(tc11)*ctcv11
wacal=np.cos(tcal)*ctcual + np.sin(tcal)*ctcval

mt2=waoal

smin =  0.5
smax =  3.0
swdt = 0.5
idv = np.int((smax-smin)/swdt)+1
tmax= 180.
tmin=  90.
twdt= 10
aaa='amplitude/(cm s-1)'
bbb='phase/degree'
arng=np.array((smin-0.000,smax+0.3))
prng=np.array((tmin-1.5,tmax+1.5))
athc=np.array((smin,smax,swdt))
pthc=np.array((tmin,tmax,twdt))
refstd=1.0
refft=waoal
fig = plt.figure(figsize=(6,6))
dia = AmpPhsDiagram(reff=refft,fig=fig,amprange=arng,ampthck=athc,phsrange=prng,phsthck=pthc,amptitle=aaa,phstitle=bbb)
contours=dia.bdd_contours(levs=[0.5,1.0,1.5],colors='0.5')
plt.clabel(contours, inline=1, fontsize=10,fmt="%.1f")
plt.scatter(waoal.real,waoal.imag,color='tomato',marker='*', label="Obs.",s=240)
plt.scatter(wacal.real,wacal.imag, color='dodgerblue', marker="s",label="Model",s=100)
omy=waoal/np.abs(waoal)
plt.plot((omy.real*0.5,omy.real*4.),(omy.imag*0.5,omy.imag*4.),color='tomato')
cmy=wacal/np.abs(wacal)
plt.plot((cmy.real*0.5,cmy.real*4.),(cmy.imag*0.5,cmy.imag*4.),color='dodgerblue')
for ii in np.arange(np.arctan2(omy.imag,omy.real),np.arctan2(cmy.imag,cmy.real)+0.01,0.030):
    plt.scatter( np.abs(waoal)*np.cos(ii),np.abs(waoal)*np.sin(ii), marker='.',color='orchid')

plt.legend(loc="upper left",ncol=3,scatterpoints=1)
plt.savefig('./SAM2comp.eps')
plt.savefig('./SAM2comp.png')
# phase lag = 26.3

smin =  0.0
smax = 3.0
swdt = 0.5
tmax=  90.
tmin= - 0.
twdt=  10
aaa = 'M2_amplitude/(cm s-1)'
bbb = 'Inclination/(degree)'
refft=wpoal*roal+wmoal/roal
arng=np.array((smin-0.0,smax+0.3))
prng=np.array((tmin-0.0,tmax+0.0))
athc=np.array((smin,smax,swdt))
pthc=np.array((tmin,tmax,twdt))

dvct=7.0/180.*np.pi
svct=aa*np.cos(dvct)+bb*np.sin(dvct)


fig = plt.figure(figsize=(6,6))
dia = AmpPhsDiagram(reff=refft,fig=fig,amprange=arng,ampthck=athc,phsrange=prng,phsthck=pthc,amptitle=aaa,phstitle=bbb)
contours=dia.bdd_contours(levs=[0.5,1.0,1.5],colors='0.5')
plt.clabel(contours, inline=1, fontsize=10,fmt="%.1f")
dmy=wpoal*roal
plt.scatter(dmy.real,dmy.imag,edgecolor='tomato',marker='o', label="Obs. CCV",s=100,color='darkgray')
dmy=wpcal*roal
vmy=dmy*svct-dmy
plt.quiver(dmy.real, dmy.imag, vmy.real,vmy.imag,scale_units='xy', scale=1.,color='k',width=0.006)
plt.scatter(dmy.real,  dmy.imag, edgecolor='dodgerblue', marker="o",label="Model CCV",s=100,color='darkgray')
pcmy=dmy

dmy=wmoal/roal
plt.scatter(dmy.real,dmy.imag,edgecolor='tomato',marker='^', label="Obs. CV",s= 90,color='white')
dmy=wmcal/roal
vmy=dmy/svct-dmy
plt.quiver(dmy.real, dmy.imag, vmy.real,vmy.imag,scale_units='xy', scale=1.,color='orchid',width=0.006)
plt.scatter(dmy.real,  dmy.imag, edgecolor='dodgerblue', marker="^",label="Model CV",s=90,color='white')

dmy=wpoal*roal+wmoal/roal
plt.scatter(dmy.real,dmy.imag,color='tomato',marker='*', label="Obs. MA",s=150,edgecolor='tomato')
dmy=dmy/np.abs(dmy)
plt.plot((dmy.real*0.0,dmy.real*4.),(dmy.imag*0.0,dmy.imag*4.),color='tomato')
emy=dmy
dmy=wpcal*rcal+wmcal/rcal
plt.scatter(dmy.real,  dmy.imag, edgecolor='dodgerblue', marker="s",label="Model MA",s= 80,color='dodgerblue')
dmy=dmy/np.abs(dmy)
plt.plot((dmy.real*0.0,dmy.real*4.),(dmy.imag*0.0,dmy.imag*4.),color='dodgerblue')
acmy=dmy
dia=np.abs(wpoal)+np.abs(wmoal)-0.
print (np.arctan2(emy.imag,emy.real)-np.arctan2(acmy.imag,acmy.real)/np.pi*180)
#-15.2
print (np.abs(wpoal)+np.abs(wmoal) )# 2.09
print (np.abs(wpoal),np.abs(wmoal) )#1.05,1.04

print (np.abs(wpcal)+np.abs(wmcal) )#3.11 
print (np.abs(wpcal),np.abs(wmcal) )#1.57,1.55

for ii in np.arange(np.arctan2(emy.imag,emy.real),np.arctan2(acmy.imag,acmy.real)+0.03,0.030):
    plt.scatter( dia*np.cos(ii),dia*np.sin(ii), marker='.',color='gray')

dia=np.abs(pcmy)
for ii in np.arange(np.arctan2(pcmy.imag,pcmy.real),np.arctan2(acmy.imag,acmy.real)+0.01,0.030):
    plt.scatter( dia*np.cos(ii),dia*np.sin(ii), marker='.',color='orchid',s=10)

plt.legend(loc="upper center",ncol=3,scatterpoints=1)
dmy=wpoal*roal+wmoal/roal
plt.scatter(dmy.real,dmy.imag,color='tomato',marker='*', s=150,edgecolor='tomato')
dmy=wpcal*roal
vmy=dmy*svct-dmy
plt.quiver(dmy.real, dmy.imag, vmy.real,vmy.imag,scale_units='xy', scale=1.,color='orchid',width=0.006)
plt.scatter(dmy.real,  dmy.imag, edgecolor='dodgerblue', marker="o",s=100,color='darkgray')
#plt.scatter(dmy.real,  dmy.imag, edgecolor='dodgerblue', marker="*",s=150,color='white')
plt.savefig('./VdaenM2comp.eps')
plt.savefig('./VdaenM2comp.png')






