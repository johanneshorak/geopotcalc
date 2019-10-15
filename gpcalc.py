import xarray as xa
import numpy as np
import pandas as pd

# # ---------------------------------------------------------------------------------
# library to calculate the geopotential for each grid cell / model level
# details are documented here:
# https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
# # ---------------------------------------------------------------------------------

era_ds  = None
ml_df   = None
lvlmax  = None # 61 for ERAI, 137 for ERA5. set by calling set_data() function
source  = None

# ---------------------------------------------------------------------------------
# some gas constants
# source: http://glossary.ametsoc.org/wiki/Gas_constant
# ---------------------------------------------------------------------------------
R_D = 287.06    # dry air,     J K**-1 kg**-1
R_V = 461       # water vapor, J K**-1 kg**-1

# ---------------------------------------------------------------------------------
# source:  IFS Documentation - Cy41r1, Part III: Dynamics and Numerical Procedures
# 
# all equations referenced are found in this document
# ---------------------------------------------------------------------------------

# ----------------------------------------------
# function set_data
# ----------------------------------------------
# set data thats required for the calculations here.
def set_data(eds, mldf, lvls):
    global era_ds
    global ml_df
    global lvlmax
    global source
    
    era_ds  = eds
    ml_df   = mldf
    lvlmax  = lvls
    
    if lvls == 137:
        source = 'era5'
    elif lvls == 60:
        source = 'erai'
    else:
        source = 'error'
        print('WARNING!')
        print('unknown data source, number of model levels supplied not associated with')
        print('either erai or era5!')

# ----------------------------------------------
# function get_phikhalf
# ----------------------------------------------
# implementation of equation 2.21
# calculates geopotential at half layer k+1/2
# for half layer 137+1/2 this equals the geopotential at the surface
#
# parameters
#
# k      full level below which to calculate phi_(k+1/2) (half layer k+1/2)
#
# concerning half and full levels/layers:
#   (k+1/2 lies below full layer k)
# 
#   level 0.5 corresponds to the top of the atmosphere
#   level 137.5 to the surface
#   full levels range from 1 to 137
# ----------------------------------------------
def get_phikhalf(k):
    global era_ds
    global lvlmax

    s  = 0                  # value of sum
    sp = era_ds.sp.values  # surface pressure
    
    #print(k,' loop from ',k+1,' to ',lvlmax)
    
    for j in range(k+1,lvlmax+1):    
        #print ('   loop ',j)
        pu = get_p(j,-0.5)                    # pressure at half layer above (lower than pl)
        pl = get_p(j,+0.5)                    # pressure at half layer below (higher than pu)
        
        t  = era_ds.sel(level=j)['t'].values # temperature at full level j
        q  = era_ds.sel(level=j)['q'].values # specific humidity at full level j
        tv = t*(1+(R_V/R_D-1.0)*q)            # virtual temperature at full level j
        
        s+=R_D*tv*np.log(pl/pu)               # sum in eq. 2.21
        
    s += era_ds.z.values                     # need to add geopotential at surface
    return s

# ----------------------------------------------
# function get_p
# ----------------------------------------------
# get pressure at half level
#
# k  ... full level
# hl ... which half level
#
# hl = +0.5 ... half level below (k+1/2)
# hl = -0.5 ... half level above (k-1/2)
# ----------------------------------------------
def get_p(k,hl):
    global era_ds
    global ml_df
    global lvlmax
    global source
    
    # adressing of the half levels differs slightly between era5 and erai
    # that's why this case distinction is necessary.
    if source == 'era5':
        if hl == 0.5:      # half layer below full level k
            h = 0
        elif hl == -0.5:   # half layer above full level k
            h = -1
        else:
            print('error, hl needs to be +0.5 or -0.5')
    elif source == 'erai': # for erai this is slightly different in how the levels are numbered
        if hl == 0.5:      # half layer below full level k
            h = 1
        elif hl == -0.5:   # half layer above full level k
            h = 0
        else:
            print('error, hl needs to be +0.5 or -0.5')
            
    p = era_ds.sp.values*np.nan
    if (k < lvlmax) or ((k == lvlmax) and (h==-1) and (source=='era5')) or ((k == lvlmax) and (h==0) and (source=='erai')):
        a = ml_df.loc[k+h,'a [Pa]']
        b = ml_df.loc[k+h,'b']
        p = a+b*era_ds.sp.values
    elif ((source == 'era5') and (k == lvlmax) and(h == 0)) or ((source == 'erai') and (k == lvlmax) and(h == 1)):
        # if the half level below 137 or 60 is requested that's the surface pressure (erai numbering is slightly different)
        p = era_ds.sp.values
        a = np.nan
        b = np.nan
    #print(k,' ',h, ' ',lvlmax)
    #print(' k=',k, ' hl=',hl,' h=',h,' p=',p[0,0,0],' a=', a,' b=', b)    
    return p

# ----------------------------------------------
# function get_alpha
# ----------------------------------------------
# calculate coefficient alpha_k
# as given by equation 2.23
# ----------------------------------------------
def get_alpha(k):   
    if k == 1:
        ak = np.log(2)
    else:
        pu = get_p(k,-0.5)
        pl = get_p(k,+0.5)
        
        deltapk = pl-pu
        ak = 1-(pu/deltapk)*np.log(pl/pu)
    return ak
  
    
    
# ----------------------------------------------
# function get_phi
# ----------------------------------------------
# calculate geopotential at full model level k
# as given by equation 2.22
# ----------------------------------------------
def get_phi(k):
    global era_ds

    phikhl = get_phikhalf(k)
    ak    = get_alpha(k)

    t  = era_ds.sel(level=k)['t'] # temperature at full level k
    q  = era_ds.sel(level=k)['q'] # specific humidity at full level k
    tv = t*(1+(R_V/R_D-1.0)*q)     # virtual temperature at full level k
    
    phik = phikhl+ak*R_D*tv
    
    return phik
