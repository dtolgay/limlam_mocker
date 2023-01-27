import numpy as np
import fast_histogram
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from astropy.convolution import Gaussian2DKernel, convolve
from ..tools import empty_table, timeme
from .. import debug
from .. import I_line, T_line


@timeme
def halo_attrcut_subset(halos, attr, minval, maxval):
    """
    crops the halo catalogue to only include desired halos
    """
    subset = empty_table()
    dm = [(getattr(halos,attr) > minval) * (getattr(halos,attr) <= maxval)]

    for i in dir(halos):
        if i[0]=='_': continue
        try:
            setattr(subset,i,getattr(halos,i)[dm])
        except TypeError:
            pass
    subset.nhalo = len(subset.M)
    subset.output_file = halos.output_file
    if debug.verbose: print('\n\t%d halos remain after attribute cut' % subset.nhalo)
    return subset

def halo_masscut_subset(halos, min_mass, max_mass):
    return halo_attrcut_subset(halos, 'M', min_mass, max_mass)

def halo_vmaxcut_subset(halos, min_vmax, max_vmax):
    return halo_attrcut_subset(halos, 'vmax', min_vmax, max_vmax)

def Lco_to_map_doppler(halos,mapinst,units='temperature',
                        binattr='vmax',bincount=100,fwhmfunc=None,fwhmattr=None,freqrefine=10,
                        filterfunc=gaussian_filter1d,lazyfilter=True):
    if binattr not in dir(halos):
        raise AttributeError('binattr {} not in dir(halos)'.format(binattr))
    ### Calculate line freq from redshift
    halos.nu  = mapinst.nu_rest/(halos.redshift+1)

    # Transform from Luminosity to Temperature (uK)
    # ... or to flux density (Jy/sr)
    if (units=='intensity'):
        if debug.verbose: print('\n\tcalculating halo intensities')
        halos.Tco = I_line(halos, mapinst)
    else:
        if debug.verbose: print('\n\tcalculating halo temperatures')
        halos.Tco = T_line(halos, mapinst)

    if 1==bincount:
        subsets = [halos]
    else:
        binattr_val = getattr(halos,binattr)
        attr_ranges = np.linspace(min(binattr_val)*(1-1e-16),max(binattr_val), bincount+1)
        subsets = [halo_attrcut_subset(halos,binattr,v1,v2)
                        for v1,v2 in zip(attr_ranges[:-1],attr_ranges[1:])]

    # flip frequency bins because np.histogram needs increasing bins
    bins3D_fine = [mapinst.pix_binedges_x, mapinst.pix_binedges_y,
                        np.linspace(min(mapinst.nu_binedges),
                                    max(mapinst.nu_binedges),
                                    len(mapinst.nu_binedges[1:])*freqrefine+1)]
    dnu_fine = np.mean(np.diff(bins3D_fine[-1]))

    # bin in RA, DEC, NU_obs
    maps = np.zeros((len(mapinst.pix_bincents_x),len(mapinst.pix_bincents_y),len(mapinst.nu_bincents)))
    if fwhmfunc is None:
        # a fwhmfunc is needed to turn halo attributes into a line width
        #   (in observed frequency space)
        # default fwhmfunc based on halos is vmax/c times observed frequency
        if fwhmattr is not None:
            fwhmfunc = lambda h:h.nu*getattr(h,fwhmattr)/299792.458
        else:
            fwhmfunc = lambda h:h.nu*h.vmax/299792.458
    for i,sub in enumerate(subsets):
        if sub.nhalo < 1: continue;
        maps_fine = np.histogramdd( np.c_[sub.ra, sub.dec, sub.nu],
                                      bins    = bins3D_fine,
                                      weights = sub.Tco )[0]
        if callable(fwhmfunc):
            sigma = 0.4246609*np.nanmedian(fwhmfunc(sub)) # in freq units (GHz)
        else: # hope it's a number
            sigma = 0.4246609*fwhmfunc # in freq units (GHz)
        if sigma > 0:
            if lazyfilter:
                if lazyfilter=='rehist':
                    # uses fast_histogram assuming mapinst bins are evenly spaced
                    filteridx = fast_histogram.histogram2d(sub.ra,sub.dec,
                                                (mapinst.npix_x, mapinst.npix_y),
                                                ((-mapinst.fov_x/2,mapinst.fov_x/2),
                                                 (-mapinst.fov_y/2,mapinst.fov_y/2)))>0
                else:
                    filteridx = np.where(np.any(maps_fine,axis=-1))
                maps_fine[filteridx] = filterfunc(maps_fine[filteridx],sigma/dnu_fine)
            else:
                maps_fine = filterfunc(maps_fine,sigma/dnu_fine)
        maps+= np.sum(maps_fine.reshape((maps_fine.shape[0],maps_fine.shape[1],-1,freqrefine)),axis=-1)
        if debug.verbose:
            print('\n\tsubset {} / {} complete'.format(i,len(subsets)))
    if (units=='intensity'):
        maps/= mapinst.Ompix
    # flip back frequency bins
    return maps[:,:,::-1]

# dd 30.06.2022
def Lco_to_map_doppler_synthbeam(halos,mapinst,units='temperature',
                        binattr='vmax',bincount=100,fwhmfunc=None,fwhmattr=None,freqrefine=10,
                        filterfunc=gaussian_filter1d,lazyfilter=True,
                        beamfwhm=4.5,xbins=120,xrefine=10):
    """
    makes a 3D LIM map accounting for observational effects both spatially and spectrally:
    will use chung+21's simplest method for simulating line broadening in the galaxies
    (with some simulated random inclination) and then also smear in the spatial axes by
    a gaussian beam with some fwhm
    beamfwhm is in arcminutes
    """
    # SPATIAL BROADENING SETUP
    if binattr not in dir(halos):
        raise AttributeError('binattr {} not in dir(halos)'.format(binattr))
    ### Calculate line freq from redshift
    halos.nu  = mapinst.nu_rest/(halos.redshift+1)

    # Transform from Luminosity to Temperature (uK)
    # ... or to flux density (Jy/sr)
    if (units=='intensity'):
        if debug.verbose: print('\n\tcalculating halo intensities')
        halos.Tco = I_line(halos, mapinst)
    else:
        if debug.verbose: print('\n\tcalculating halo temperatures')
        halos.Tco = T_line(halos, mapinst)

    # BINS HALOS SPECTRALLY BY VVIR, MAKES MAPS OF EACH SUBSET, SMOOTHS THEM, AND
    # THEN COMBINES
    if 1==bincount:
        subsets = [halos]
    else:
        binattr_val = getattr(halos,binattr)
        attr_ranges = np.linspace(min(binattr_val)*(1-1e-16),max(binattr_val), bincount+1)
        subsets = [halo_attrcut_subset(halos,binattr,v1,v2)
                        for v1,v2 in zip(attr_ranges[:-1],attr_ranges[1:])]

    # SET UP FINER BINNING IN RA, DEC, FREQUENCY
    # flip frequency bins because np.histogram needs increasing bins
    bins3D_fine = [np.linspace(min(mapinst.pix_binedges_x),
                               max(mapinst.pix_binedges_x),
                               len(mapinst.pix_binedges_x[1:])*xrefine+1),
                   np.linspace(min(mapinst.pix_binedges_y),
                               max(mapinst.pix_binedges_y),
                               len(mapinst.pix_binedges_y[1:])*xrefine+1),
                    np.linspace(min(mapinst.nu_binedges),
                                max(mapinst.nu_binedges),
                                len(mapinst.nu_binedges[1:])*freqrefine+1)]
    dx_fine = np.mean(np.diff(bins3D_fine[0]))
    dy_fine = np.mean(np.diff(bins3D_fine[1]))
    dnu_fine = np.mean(np.diff(bins3D_fine[-1]))

    maps = np.zeros((len(mapinst.pix_bincents_x)*xrefine,
                     len(mapinst.pix_bincents_y)*xrefine,
                     len(mapinst.nu_bincents)))

    # bin in (FINER) RA, DEC, NU_obs
    if fwhmfunc is None:
        # a fwhmfunc is needed to turn halo attributes into a line width
        #   (in observed frequency space)
        # default fwhmfunc based on halos is vmax/c times observed frequency
        if fwhmattr is not None:
            fwhmfunc = lambda h:h.nu*getattr(h,fwhmattr)/299792.458
        else:
            fwhmfunc = lambda h:h.nu*h.vmax/299792.458
    for i,sub in enumerate(subsets):
        if sub.nhalo < 1: continue;
        maps_fine = np.histogramdd( np.c_[sub.ra, sub.dec, sub.nu],
                                      bins    = bins3D_fine,
                                      weights = sub.Tco )[0]
        if callable(fwhmfunc):
            sigma = 0.4246609*np.nanmedian(fwhmfunc(sub)) # in freq units (GHz)
        else: # hope it's a number
            sigma = 0.4246609*fwhmfunc # in freq units (GHz)
        if sigma > 0:
            if lazyfilter:
                if lazyfilter=='rehist':
                    # uses fast_histogram assuming mapinst bins are evenly spaced
                    filteridx = fast_histogram.histogram2d(sub.ra,sub.dec,
                                                (mapinst.npix_x, mapinst.npix_y),
                                                ((-mapinst.fov_x/2,mapinst.fov_x/2),
                                                 (-mapinst.fov_y/2,mapinst.fov_y/2)))>0
                else:
                    filteridx = np.where(np.any(maps_fine,axis=-1))
                maps_fine[filteridx] = filterfunc(maps_fine[filteridx],sigma/dnu_fine)
            else:
                maps_fine = filterfunc(maps_fine,sigma/dnu_fine)

        maps+= np.sum(maps_fine.reshape((maps_fine.shape[0],maps_fine.shape[1],-1,freqrefine)),axis=-1)
        if debug.verbose:
            print('\n\tsubset {} / {} complete'.format(i,len(subsets)))

    # at this point have a doppler-broadened map with science spectral resolution and the refined
    # spatial resolution. need to beam-smear and then rebin in space

    # number of refined pixels corresponding to the fwhm in arcminutes
    std = beamfwhm / (2*np.sqrt(2*np.log(2))) / 60 # standard deviation in degrees
    std_pix = std / dx_fine

    # COMAP spatial synthesized beam
    beamkernel = Gaussian2DKernel(std_pix)
    # loop over the frequency axis and convolve each frame
    smoothsimlist = []

    if debug.verbose:
        print('\nsmoothing by synthesized beam: {} channels total'.format(maps.shape[-1]))
    for i in range(maps.shape[-1]):
        smoothsimlist.append(convolve(maps[:,:,i], beamkernel))
        if debug.verbose:
            if i%100 == 0:
                print('\n\t done {} of {} channels'.format(i, maps.shape[-1]))

    maps_sm_fine = np.stack(smoothsimlist, axis=-1)

    # rebin
    mapssm = np.sum(maps_sm_fine.reshape((len(mapinst.pix_bincents_x), xrefine,
                                          len(mapinst.pix_bincents_y), xrefine, -1)), axis=(1,3))

    if (units=='intensity'):
        mapssm/= mapinst.Ompix
    # flip back frequency bins
    return mapssm[:,:,::-1]


def Lco_to_map_doppler_vvirfast(halos,mapinst,units='temperature',
                        bincount=100,vvirfunc=None,freqrefine=10):
    if ('vvir' not in dir(halos)):
        if (not callable(vvirfunc)):
            raise AttributeError('vvir not in dir(halos) (and no callable vvirfunc specified)')
        else:
            halos.vvir = vvirfunc(halos)
    ### Calculate line freq from redshift
    halos.nu  = mapinst.nu_rest/(halos.redshift+1)

    # Transform from Luminosity to Temperature (uK)
    # ... or to flux density (Jy/sr)
    if (units=='intensity'):
        if debug.verbose: print('\n\tcalculating halo intensities')
        halos.Tco = I_line(halos, mapinst)
    else:
        if debug.verbose: print('\n\tcalculating halo temperatures')
        halos.Tco = T_line(halos, mapinst)

    vvir_ranges = np.linspace(min(halos.vvir)*(1-1e-16),max(halos.vvir), bincount+1)
    vvir_centvals = (vvir_ranges[1:]+vvir_ranges[:-1])/2
    # flip frequency bins because np.histogram needs increasing bins
    bins_fine = [vvir_ranges, mapinst.pix_binedges_x, mapinst.pix_binedges_y,
                        np.linspace(min(mapinst.nu_binedges),
                                    max(mapinst.nu_binedges),
                                    len(mapinst.nu_binedges[1:])*freqrefine+1)]
    dnu_fine = np.mean(np.diff(bins_fine[-1]))

    # bin in RA, DEC, NU_obs
    maps_fine = np.histogramdd(np.c_[halos.vvir, halos.ra, halos.dec, halos.nu],
                                      bins    = bins_fine,
                                      weights = halos.Tco )[0]
    sigmas = np.median(mapinst.nu_bincents)*vvir_centvals/299792.458*0.4246609
    for i,(m,s) in enumerate(zip(maps_fine,sigmas)):
        filteridx = np.any(m,axis=-1)
        d = int(np.floor(s/dnu_fine*1.88+0.5))
        m[filteridx] = uniform_filter1d(m[filteridx], d)
        m[filteridx] = uniform_filter1d(m[filteridx], d, origin=-(d%2==0))
        m[filteridx] = uniform_filter1d(m[filteridx], d+(d%2==0))
        maps_fine[i] = m
    maps_fine = np.sum(maps_fine,axis=0)
    maps = np.sum(maps_fine.reshape((maps_fine.shape[0],maps_fine.shape[1],
                                                    -1,freqrefine)),axis=-1)
    if (units=='intensity'):
        maps/= mapinst.Ompix
    # flip back frequency bins
    return maps[:,:,::-1]
