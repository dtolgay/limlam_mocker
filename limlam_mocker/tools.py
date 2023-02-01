from __future__ import print_function
from . import debug
import time
import datetime
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy 

class empty_table():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)

def write_time(string_in):
    """
    write time info in as nicely formatted string
    """
    fmt       = '%H:%M:%S on %m/%d/%Y'
    timestamp = datetime.datetime.now().strftime(fmt)
    bar = 72*'-'
    print( '\n\n'+bar )
    print( string_in )
    print( 'Time:      '+timestamp )
    print( bar+'\n' )

    return

def timeme(method):
    """
    writes the time it takes to run a function
    To use, pput above a function definition. eg:
    @timeme
    def Lco_to_map(halos,map):
    """
    def wrapper(*args, **kw):
        startTime = int(round(time.time()))
        result = method(*args, **kw)
        endTime = int(round(time.time()))

        if debug.verbose: print('  ',endTime - startTime,'sec')
        return result

    return wrapper

def module_directory(name_module, path):
    """
    Allows for modules to be imported to be passed as strings, and to be
    updated dynamically.

    Inputs:
    -------
    name_module : string
        file name to be imported as a module
    path: path to the module file to be imported (INCLUDING file name)

    """
    P = importlib.util.spec_from_file_location(name_module, path)
    import_module = importlib.util.module_from_spec(P)
    P.loader.exec_module(import_module)
    return import_module

def make_output_filenames(params, outputdir=None):
    """
    Uses the parameters in the input file to automatically change the name of the
    result files to be output - the cube file and the two plot files if plotting
    """
    # default to a folder with the model name in a folder called output
    if not outputdir:
        outputdir = './output/' + params.model

    # make the output directory if it doesn't already exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    halofile = params.halo_catalogue_file
    seedname = halofile[halofile.find('seed'):-4]

    params.map_output_file = outputdir + '/Lco_cube_' + params.model + '_' + seedname
    params.halo_output_file = outputdir + '/Lco_cat_' + params.model + '_' + seedname
    params.plot_cube_file = outputdir + '/cube_' + params.model + '_' + seedname
    params.plot_pspec_file = outputdir + '/pspec_' + params.model + '_' + seedname
    return

def params_to_mapinst(params):
    """
    Adds input parameters to be kept by the map class and gets map details

    Returns
    -------
    map : class
       contains all information about the map that the halos will be binned into
    """
    map             = empty_table() # creates empty class to put map info into

    map.output_file = params.map_output_file

    map.nmaps  = int(params.nmaps)
    map.fov_x  = float(params.fov_x)
    map.fov_y  = float(params.fov_y)
    map.npix_x = int(params.npix_x)
    map.npix_y = int(params.npix_y)
    map.nu_i   = float(params.nu_i)
    map.nu_f   = float(params.nu_f)
    map.nu_rest= float(params.nu_rest)
    map.z_i    = map.nu_rest/map.nu_i - 1
    map.z_f    = map.nu_rest/map.nu_f - 1

    # get arrays describing the final intensity map to be output
    # map sky angle dimension
    map.pix_size_x = map.fov_x/map.npix_x
    map.pix_size_y = map.fov_y/map.npix_y

    # pixel size to convert to brightness temp
    map.Ompix = (map.pix_size_x*np.pi/180)*(map.pix_size_y*np.pi/180)

    map.pix_binedges_x = np.linspace(-map.fov_x/2,map.fov_x/2,map.npix_x+1)
    map.pix_binedges_y = np.linspace(-map.fov_y/2,map.fov_y/2,map.npix_y+1)

    map.pix_bincents_x =  0.5*(map.pix_binedges_x[1:] + map.pix_binedges_x[:-1])
    map.pix_bincents_y =  0.5*(map.pix_binedges_y[1:] + map.pix_binedges_y[:-1])

    # map frequency dimension
    # use linspace to ensure nmaps channels
    map.nu_binedges = np.linspace(map.nu_i,map.nu_f,map.nmaps+1)
    map.dnu         = np.abs(np.mean(np.diff(map.nu_binedges)))
    map.nu_bincents = map.nu_binedges[:-1] - map.dnu/2

    return map



# Cosmology Functions
# Explicitily defined here instead of using something like astropy
# in order for ease of use on any machine
def hubble(z,h,omegam):
    """
    H(z) in units of km/s
    """
    return h*100*np.sqrt(omegam*(1+z)**3+1-omegam)

def drdz(z,h,omegam):
    return 299792.458 / hubble(z,h,omegam)

def chi_to_redshift(chi, cosmo):
    """
    Transform from redshift to comoving distance
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    chiinterp -= chiinterp[0]
    z_of_chi   = sp.interpolate.interp1d(chiinterp,zinterp)

    return z_of_chi(chi)

def redshift_to_chi(z, cosmo):
    """
    Transform from comoving distance to redshift
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    chiinterp -= chiinterp[0]
    chi_of_z   = sp.interpolate.interp1d(zinterp,chiinterp)

    return chi_of_z(z)


def plot_results(mapinst,k,Pk,Pk_sampleerr,params):
    """
    Plot central frequency map and or powerspectrum
    """
    if debug.verbose: print("\n\tPlotting results")

    ### Plot central frequency map
    plt.rcParams['font.size'] = 16
    if params.plot_cube:
        plt.figure().set_tight_layout(True)
        # im = plt.imshow(np.log10(mapinst.maps[:,:,params.nmaps//2]+1e-6), extent=[-mapinst.fov_x/2,mapinst.fov_x/2,-mapinst.fov_y/2,mapinst.fov_y/2],vmin=-1,vmax=2)
        im = plt.imshow(np.log10(mapinst.maps[:,:,params.nmaps//2]+1e-6), extent=[-mapinst.fov_x/2,mapinst.fov_x/2,-mapinst.fov_y/2,mapinst.fov_y/2], vmin=-6,vmax=15)
        # plt.colorbar(im,label=r'$log_{10}\ T_b\ [\mu K]$')
        plt.colorbar(im,label=r'$log_{10}\ L_☉$')
        plt.xlabel('degrees',fontsize=20)
        plt.ylabel('degrees',fontsize=20)
        plt.title('simulated map at {0:.3f} GHz'.format(mapinst.nu_bincents[params.nmaps//2]),fontsize=24)
        # plt.savefig(params.plot_cube_file)

    if params.plot_pspec:
        plt.figure().set_tight_layout(True)
        plt.errorbar(k,k**3*Pk/(2*np.pi**2),k**3*Pk_sampleerr/(2*np.pi**2),
                     lw=3,capsize=0)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(True)
        plt.xlabel('k [1/Mpc]',fontsize=18)
        plt.ylabel('$\\Delta^2(k)$ [$\\mu$K$^2$]',fontsize=18)
        plt.title('simulated line power spectrum',fontsize=24)
        # plt.savefig(params.plot_pspec_file)

    if params.plot_cube or params.plot_pspec:
        plt.show()
        backpath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        os.chdir(backpath)

    return
