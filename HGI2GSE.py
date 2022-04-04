#!/usr/bin/env python3

'''
    File Name: HGI2GSE.py
    Author: Yingjie Zhu
    Institute: University of Michigan
    Email: yjzhu(at)umich(dot)edu
    Date create: 15/03/2022
    Date Last Modified: 15/03/2022
    Python Version: 3.7
    Version: 0.0.1
    Description: This is a naive and dirty python script 
    to transform the vectors in IH Satellite files from 
    Heliographic/Heliocentric Inertial coordinates (HGI/HCI) 
    to Geocentric Solar Ecliptic (GSE) coordinates. Note that 
    the transformation is actually not self-consistent, especially 
    the 1.4 deg/century precession of the longitude of ascending node
    between ecliptic and solar equator is not taken into account. 
'''

import numpy as np
import scipy 
from scipy.spatial.transform import Rotation as R
import astropy.constants as const
import datetime


'''
    Rotate the velocity and field vector at Earth from HGI/HCI into GSE. 
    Use a approximated motion of Earth vy = 29.7859 in GSE. 
    Parameters
    ----------
    x_earth : float
    X coordinate of Earth in HGI/HCI. 
    y_earth : float
    Y coordinate of Earth in HGI/HCI.
    z_earth : float
    Z coordinate of Earth in HGI/HCI.
    ux : float
    X component of velocity in HGI/HCI.
    uy : float
    Y component of velocity in HGI/HCI.
    uz : float
    Z component of velocity in HGI/HCI.
    bx : float
    X component of magnetic field vector in HGI/HCI.
    by : float
    X component of magnetic field vector in HGI/HCI.
    bz : float
    X component of magnetic field vector in HGI/HCI.
    Returns
    -------
    u_rot: 3-element array
    The transformed velocity vector in the GSE coordinate.
    b_rot: 3-element array
    The transformed field vector in the GSE coordinate.


'''
def convert_HGI_GSE_approx(x_earth,y_earth,z_earth,
                            ux,uy,uz,bx,by,bz):
    earth_coord = np.array([x_earth, y_earth, z_earth])

    u_vector = np.array([ux,uy,uz])
    b_vector = np.array([bx,by,bz])

    # mean obital velocity of Earth from F&H 2002, add it to uy
    # in GSE to approximate the relative motion of Earth
    v0 = 29.7859
    # Earth-Sun line as vector X_GSE 
    xprime_gse = -earth_coord
    xprime_gse = xprime_gse/np.linalg.norm(xprime_gse)

    # Use X_HGI x X_GSE to evaluate the normal vector of the ecliptic (Z_GSE). 
    # It is not 100% accurate, because X_HGI/X_HCI is the intersection line 
    # of the solar ecliptic and equator on J1900/J2000. Due to the 
    # 1.4 deg/century precession of the longitude of ascending node, 
    # the X_HGI/X_HCI no longer perfectly lies in the ecliptic.
  
    zprime_gse = np.cross(np.array([1,0,0]),xprime_gse)
    zprime_gse = zprime_gse/np.linalg.norm(zprime_gse)
    if zprime_gse[2] < 0:  # make sure Z_GSE points to the north
        zprime_gse = -zprime_gse

    # finish Y_GSE with the right hand rule
    yprime_gse = np.cross(zprime_gse, xprime_gse)

    # the passive rotation matrix
    rot_matrix = np.row_stack((xprime_gse,yprime_gse,zprime_gse))
    rotation = R.from_matrix(rot_matrix)

    u_rot = rotation.apply(u_vector) + np.array([0, v0, 0])
    b_rot = rotation.apply(b_vector)

    return u_rot, b_rot
    
'''
    Rotate a vector at Earth from HGI/HCI into GSE.

    Parameters
    ----------
    earth_coord : 3-element array
    HGI/HCI coordinates of Earth in solar radii.
    date : datetime.datetime obj
    Time of observation. Can be created from BATSRUS satellite files by 
    datetime.datetime(year,mo,dy,hr,mn,sc,msc)
    vector : 3-element array
    The vector (magnetic field, velocity...) to be rotated.
    earth_vel : 3-element array, optional
    If provided, use this relative velocity in velocity transformation. 
    If not provided, use the date and position of Earth to calculate 
    the velocity of Earth used in velocity transformation. Default is None.
    is_velocity : bool
    If True, the transformation will take the relative motion of Earth into 
    account. Should be only used when vector represents velocity. Default 
    is False.

    Returns
    -------
    vector_rot: 3-element array
    The transformed vector in the GSE coordinate.


'''
def convert_HGI_GSE(earth_coord,date,vector,earth_vel=None,is_velocity=False):
    # convert list input into numpy arrays
    if type(earth_coord) is list:
        earth_coord = np.array(earth_coord)
    if type(vector) is list:
        vector = np.array(vector)
    if type(earth_vel) is list:
        earth_vel = np.array(earth_vel)

    # Earth-Sun line as vector X_GSE 
    xprime_gse = -earth_coord
    xprime_gse = xprime_gse/np.linalg.norm(xprime_gse)

    # Use X_HGI x X_GSE to evaluate the normal vector of the ecliptic (Z_GSE). 
    # It is not 100% accurate, because X_HGI/X_HCI is the intersection line 
    # of the solar ecliptic and equator on J1900/J2000. Due to the 
    # 1.4 deg/century precession of the longitude of the ascening node, 
    # the X_HGI/X_HCI no longer perfectly lies in the ecliptic.
  
    zprime_gse = np.cross(np.array([1,0,0]),xprime_gse)
    zprime_gse = zprime_gse/np.linalg.norm(zprime_gse)
    if zprime_gse[2] < 0:  # make sure Z_GSE points to the north
        zprime_gse = -zprime_gse

    # finish Y_GSE with the right hand rule
    yprime_gse = np.cross(zprime_gse, xprime_gse)

    # the passive rotation matrix
    rot_matrix = np.row_stack((xprime_gse,yprime_gse,zprime_gse))
    rotation = R.from_matrix(rot_matrix)

    if is_velocity:  # remove relative motion if the vector is velocity
        if earth_vel is None:  # if Earth velocity is not provided
            vr_earth, vphi_earth = get_vr_vphi_earth(earth_coord,date)
            vector_rot = rotation.apply(vector) + np.array([vr_earth, vphi_earth, 0])
        else:
            vector_rot = rotation.apply(vector - earth_vel)
    else:
        vector_rot = rotation.apply(vector)
    
    return vector_rot

'''
    Calculate the radial (vr) and tagnetial velocity (vphi) of 
    Earth in the ecliptic plane. Assuming the conservation of 
    specific mechanical energy E = v^2/2 - G0(MSun + MEarth)/r 
    and specific angular momentum h = |rxv| (see page 11, 
    Franz & Harper, 2002, 
    https://www2.mps.mpg.de/homes/fraenz/systems/systems3art.pdf).
    The specific mechanical energy and specific angular momentum
    are calculated from the position and velocity of Earth on 
    J2000 in HAE by SolarSoft get_sunspice_coord. The direction 
    of vr is determined by compare the date with the apehelion and 
    perihelion. 

    Parameters
    ----------
    coord_HCI : 3-element array 
    The position of Earth in HCI/HGI coordinates. Units in solar radii.  
    rsun : float, optional
    The solar radii in meters. If not provided, use the solar radii in 
    astropy.constants. 

    Returns
    -------
    vr : float
    The radial velocity of Earth in km/s.
    vphi: float
    The tagential velocity of Earth in km/s.

'''
def get_vr_vphi_earth(coord_HCI,date,rsun=None):
    #It seems SWMF uses Rsun = 6.96E5 km
    if type(coord_HCI) is list:
        coord_HCI = np.array(coord_HCI)

    if rsun is None:
        rsun = const.R_sun.value


    # convert coord into SI units 
    x_earth = coord_HCI*rsun
    r_earth = np.linalg.norm(x_earth)

    GM_earth = const.GM_earth.value
    GM_sun = const.GM_sun.value

    # specific mechanical energy and specific angular momentum
    # calculated from the position and velocity of Earth on 
    # J2000 in HAE by SolarSoft get_sunspice_coord

    E = -443331792.3151136
    h = 4456247350402224.0

    vphi = h/r_earth
    vr = np.sqrt(2*E + 2*(GM_sun + GM_earth)/r_earth - np.square(vphi))
    
    # dates of perihelion and apehelions to determine the sign of vr
    date_per_min = datetime.datetime(date.year,1,2)
    date_per_max = datetime.datetime(date.year,1,6)
    date_ape_min = datetime.datetime(date.year,7,3,12)
    date_ape_max = datetime.datetime(date.year,7,7,12)

    if (date < date_per_min) or (date > date_ape_max): # from apehelion to perihelion
        sign_vr = -1.
    elif (date > date_per_max) and (date < date_ape_min): # from perihelion to apehelion 
        sign_vr = 1.
    elif (date_per_min < date < date_per_max) or (date_ape_min < date < date_ape_max): 
        # close to apehelion or perihelion, vr is very small
        if np.isnan(vr):
            sign_vr = 0
            vr = 0
        elif vr < 1E2:
            sign_vr = 0 
        else:
            sign_vr = np.nan
    
    vr = vr*sign_vr

    # return in km/s
    return vr/1E3, vphi/1E3

if __name__ == "__main__":
    test_date = datetime.datetime(2005,5,14,8,10)
    test_coord = [-201.24,81.18,-10.33000]
    test_vel = [-3.61232E+02,1.44081E+02,-1.24506E+01]
    test_earth_vel = [-11.474722,-26.928305,3.4269216]
    test_b = np.array([-4.89373E-06,1.47780E-05,-2.17096E-06])*1e5

    print("Rotate velocity and field vector, only give the position of Earth.")
    print(convert_HGI_GSE_approx(*test_coord,*test_vel,*test_b))
    print(convert_HGI_GSE(test_coord,test_date,test_vel,is_velocity=True))
    print("Rotate a velocity, only give the position of Earth and date.")
    print(convert_HGI_GSE(test_coord,test_date,test_vel,is_velocity=True))
    print("Rotate a velocity, give the position of Earth, the velocity of Earth, and date.")
    print(convert_HGI_GSE(test_coord,test_date,test_vel,earth_vel=test_earth_vel,is_velocity=True))
    print("Rotate a field vector")
    print(convert_HGI_GSE(test_coord,test_date,test_b,is_velocity=False))
    # print(get_vr_vphi_earth([188.520,94.7804,-12.0631],datetime.datetime(2005,1,3)))
    # print(get_vr_vphi_earth([1.1892052,211.53320,-26.916844],datetime.datetime(2005,3,6)))
    # print(get_vr_vphi_earth([-8.81781,-214.752,27.3272],datetime.datetime(2005,9,6)))