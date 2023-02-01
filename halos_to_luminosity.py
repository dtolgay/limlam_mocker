import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate
import sys
import os
from . import debug
from .tools import *

sfr_interp_tab = None

@timeme

###################################################################################################################
# def Mhalo_to_Ly_alpha_luminosity(halos, model, coeffs):

#     """
#     General function to get L_ly_alpha(M_halo) given a certain model <model>
#     if adding your own model follow this structure,
#     and simply specify the model to use in the parameter file
#     will output halo luminosities in **L_sun**

#     Parameters
#     ----------
#     halos : class
#         Contains all halo information (position, redshift, etc..)
#     model : str
#         Model to use, specified in the parameter file
#     """    

#     dict = {'Santos': Mhalo_to_L_lyman_alpha_Santos,
#             }


#     if model in dict.keys():
#         return dict[model](halos, coeffs)

#     else:
#         sys.exit('\n\n\tYour lyman_alpha model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')




def Mhalo_to_L(halos, model, coeffs):
    """
    General function to get L_co(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file
    will output halo luminosities in **L_sun**

    Parameters
    ----------
    halos : class
        Contains all halo information (position, redshift, etc..)
    model : str
        Model to use, specified in the parameter file
    coeffs :
        None for default coeffs
    """
    dict = {'Li':          Mhalo_to_Lco_Li,
            'Li_sc':       Mhalo_to_Lco_Li_sigmasc,
            'Padmanabhan': Mhalo_to_Lco_Padmanabhan,
            'fiuducial':   Mhalo_to_Lco_fiuducial,
            'Yang':        Mhalo_to_Lco_Yang,
            'arbitrary':   Mhalo_to_Lco_arbitrary,

            'Santos':      Mhalo_to_L_lyman_alpha_Santos,
            }

    # This print is added by Doga
    print("You are using ", model, " model")

    if model in dict.keys():
        return dict[model](halos, coeffs)

    else:
        sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_L_lyman_alpha_Santos(halos, coeffs):

    """
    halo mass to SFR to L_lyman_alpha
    following the Santos 2004 model
    arXiv:astro-ph/0308196

    """

    if coeffs is None:
        # Power law parameters from paper
        sigma_sfr,sigma_L_lyman_alpha = (0.3, 0.3)
    else:
        sigma_sfr,sigma_lco = coeffs;
   


    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr);  

    # I have used the same value Dongwoo used for CO
    sigma_L_lyman_alpha = 0.3

    # Using eqn 19 from the paper:
    L_lyman_alpha_ergs_per_sec = 3.4e42 * halos.sfr                    # in erg/sec to L_solar

    # Converting to solar luminosity 
    ergs_per_sec_to_Lsolar = 1/3.826e33
    L_lyman_alpha = L_lyman_alpha_ergs_per_sec * ergs_per_sec_to_Lsolar   # in L_solar  

    # Adding scatter
    L_lyman_alpha      = add_log_normal_scatter(L_lyman_alpha, sigma_L_lyman_alpha, 2)

    if debug.verbose: print('\n\tMhalo to L_ly_alpha calculated')

    return L_lyman_alpha       




###################################################################################################################

# def Mhalo_to_Lco(halos, model, coeffs):
#     """
#     General function to get L_co(M_halo) given a certain model <model>
#     if adding your own model follow this structure,
#     and simply specify the model to use in the parameter file
#     will output halo luminosities in **L_sun**

#     Parameters
#     ----------
#     halos : class
#         Contains all halo information (position, redshift, etc..)
#     model : str
#         Model to use, specified in the parameter file
#     coeffs :
#         None for default coeffs
#     """
#     dict = {'Li':          Mhalo_to_Lco_Li,
#             'Li_sc':       Mhalo_to_Lco_Li_sigmasc,
#             'Padmanabhan': Mhalo_to_Lco_Padmanabhan,
#             'fiuducial':   Mhalo_to_Lco_fiuducial,
#             'Yang':        Mhalo_to_Lco_Yang,
#             'arbitrary':   Mhalo_to_Lco_arbitrary,
#             }

#     if model in dict.keys():
#         return dict[model](halos, coeffs)

#     else:
#         sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_Lco_Li(halos, coeffs):
    """
    halo mass to SFR to L_CO
    following the Tony li 2016 model
    arXiv 1503.08833
    """
    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = (
            0.0, 1.37,-1.74, 0.3, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop
    Lco      = add_log_normal_scatter(Lco, sigma_lco, 2)

    if debug.verbose: print('\n\tMhalo to Lco calculated')

    return Lco

def Mhalo_to_Lco_Li_sigmasc(halos, coeffs):
    """
    halo mass to SFR to L_CO
    following the Tony li 2016 model
    arXiv 1503.08833

    DD 2022 - updated to include a single lognormal scatter coeff
    """
    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sc = (
            0.0, 1.37,-1.74, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sc = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sc);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop
#    Lco      = add_log_normal_scatter(Lco, sigma_lco, 2) #DD: ?

    if debug.verbose: print('\n\tMhalo to Lco calculated')

    return Lco

def Mhalo_to_Lco_Padmanabhan(halos, coeffs):
    """
    halo mass to L_CO
    following the Padmanabhan 2017 model
    arXiv 1706.01471
    """
    if coeffs is None:
        m10,m11,n10,n11,b10,b11,y10,y11 = (
            4.17e12,-1.17,0.0033,0.04,0.95,0.48,0.66,-0.33)
    else:
        m10,m11,n10,n11,b10,b11,y10,y11 = coeffs

    z  = halos.redshift
    hm = halos.M

    m1 = 10**(np.log10(m10)+m11*z/(z+1))
    n  = n10 + n11 * z/(z+1)
    b  = b10 + b11 * z/(z+1)
    y  = y10 + y11 * z/(z+1)

    Lprime = 2 * n * hm / ( (hm/m1)**(-b) + (hm/m1)**y )
    Lco    = 4.9e-5 * Lprime

    return Lco

def Mhalo_to_Lco_fiuducial(halos, coeffs):
    """
    DD 2022, based on Chung+2022 fiuducial model
    arXiv 2111.05931
    """
    if coeffs is None:
        # default to UM+COLDz+COPSS model from Chung+22
        A, B, logC, logM, sigma = (
            -2.85, -0.42, 10.63, 12.3, 0.42)
    else:
        A,B,logC,logM,sigma = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lco = 4.9e-5 * Lprime
    Lco = add_log_normal_scatter(Lco, sigma, 3)

    return Lco

def Mhalo_to_Lco_Yang(halos, coeffs):
    """
    DD 2022, SAM from Breysse+2022/Yang+2021
    arXiv 2111.05933/2108.07716
    Not set up for anything other than CO(1-0) at COMAP redshifts currently
    becasue the model is a pretty complicated function of redshift
    for other models edit function directly with parameters from Yang+22
    """
    if coeffs is not None:
        print('The function is only set up for CO(1-0), 1<z<4')
        return 0

    z = halos.redshift
    Mh = halos.M

    # Lco function
    logM1 = 12.13 - 0.1678*z
    logN = -6.855 + 0.2366*z - 0.05013*z**2
    alpha = 1.642 + 0.1663*z - 0.03238*z**2
    beta = 1.77*np.exp(-1/2.72) - 0.00827

    M1 = 10**logM1
    N = 10**logN

    Lco = 2*N * Mh / ((Mh/M1)**(-alpha) + (Mh/M1)**(-beta))

    # fduty function
    logM2 = 11.73 + 0.6634*z
    gamma = 1.37 - 0.190*z + 0.0215*z**2

    M2 = 10**logM2

    fduty = 1 / (1 + (Mh/M2)**gamma)

    Lco = Lco * fduty

    # scatter
    sigmaco = 0.357 - 0.0701*z + 0.00621*z**2

    Lco = add_log_normal_scatter(Lco, sigmaco, 4)
    return Lco


def Mhalo_to_Lco_arbitrary(halos, coeffs):
    """
    halo mass to L_CO
    allows for utterly arbitrary models!
    coeffs:
        coeffs[0] is a function that takes halos as its only argument
        coeffs[1] is a boolean: do we need to calculate sfr or not?
        coeffs[2] is optional sigma_sfr
        coeffs[3] is optional argument that must almost never be invoked
        alternatively, if coeffs is callable, then assume we calculate sfr
            default sigma_sfr is 0.3 dex
    if sfr is calculated, it is stored as a halos attribute
    """
    sigma_sfr = 0.3
    bad_extrapolation = False
    if callable(coeffs):
        sfr_calc = True
        lco_func = coeffs
    else:
        lco_func, sfr_calc = coeffs[:2]
        if len(coeffs)>2:
            sigma_sfr = coeffs[2]
        if len(coeffs)>3:
            bad_extrapolation = coeffs[3]
    if sfr_calc:
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation)
    return lco_func(halos)

def Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation=False):
    global sfr_interp_tab
    if sfr_interp_tab is None:
        sfr_interp_tab = get_sfr_table(bad_extrapolation)
    sfr = sfr_interp_tab.ev(np.log10(halos.M), np.log10(halos.redshift+1))
    sfr = add_log_normal_scatter(sfr, sigma_sfr, 1)
    return sfr

def get_sfr_table(bad_extrapolation=False):
    """
    LOAD SFR TABLE from Behroozi+13a,b
    Columns are: z+1, logmass, logsfr, logstellarmass
    Intermediate processing of tabulated data
    with option to extrapolate to unphysical masses
    """

    tablepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tablepath+= '/tables/sfr_behroozi_release.dat'
    dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(tablepath, unpack=True)

    dat_logzp1 = np.log10(dat_zp1)
    dat_sfr    = 10.**dat_logsfr

    # Reshape arrays
    dat_logzp1  = np.unique(dat_logzp1)    # log(z), 1D
    dat_logm    = np.unique(dat_logm)    # log(Mhalo), 1D
    dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))
    dat_logsfr  = np.reshape(dat_logsfr, dat_sfr.shape)

    # optional extrapolation to masses excluded in Behroozi+13
    if bad_extrapolation:
        from scipy.interpolate import SmoothBivariateSpline
        dat_logzp1_,dat_logm_ = np.meshgrid(dat_logzp1,dat_logm)
        badspl = SmoothBivariateSpline(dat_logzp1_[-1000<(dat_logsfr)],dat_logm_[-1000<(dat_logsfr)],dat_logsfr[-1000<(dat_logsfr)],kx=4,ky=4)
        dat_sfr[dat_logsfr==-1000.] = 10**badspl(dat_logzp1,dat_logm).T[dat_logsfr==-1000.]

    # Get interpolated SFR value(s)
    sfr_interp_tab = sp.interpolate.RectBivariateSpline(
                            dat_logm, dat_logzp1, dat_sfr,
                            kx=1, ky=1)
    return sfr_interp_tab


def add_log_normal_scatter(data,dex,seed):
    """
    Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
    [via @tonyyli - https://github.com/dongwooc/imapper2]
    Note: scatter maintains mean in linear space (not log space).
    """
    if np.any(dex<=0):
        return data
    # Calculate random scalings
    sigma       = dex * 2.302585 # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
    mu          = -0.5*sigma**2

    # Set standard seed so changing minimum mass cut
    # does not change the high mass halos
    np.random.seed(seed*13579)
    randscaling = np.random.lognormal(mu, sigma, data.shape)
    xscattered  = np.where(data > 0, data*randscaling, data)

    return xscattered
