#!/usr/bin/env python
from __future__ import division
import numpy              as np
import matplotlib.pylab   as plt
import scipy              as sp
import limlam_mocker      as llm
#Get Parameters for run
import params             as params
import params_lyman_alpha as params_lyman_alpha


###########
import inspect
###########

llm.debug.verbose = True
llm.write_time('Starting Line Intensity Mapper')

### Setup maps to output
mapinst   = llm.params_to_mapinst(params);
mapinst_lyman_alpha = llm.params_to_mapinst(params_lyman_alpha);

### Load halos from catalogue
halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file)
halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst)


halos_lyman_alpha, cosmo_lyman_alpha = llm.load_peakpatch_catalogue(params_lyman_alpha.halo_catalogue_file)
halos_lyman_alpha      			     = llm.cull_peakpatch_catalogue(halos_lyman_alpha, params_lyman_alpha.min_mass, mapinst_lyman_alpha)


### Calculate Luminosity of each halo
halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs)

halos_lyman_alpha.L_lyman_alpha = llm.Mhalo_to_Ly_alpha_luminosity(halos_lyman_alpha, params_lyman_alpha.model, params_lyman_alpha.coeffs) # Change this functions name

print("inspect.getmembers(halos_lyman_alpha): ", inspect.getmembers(halos_lyman_alpha))

### Bin halo luminosities into map
mapinst.maps = llm.Lco_to_map(halos,mapinst)

mapinst_lyman_alpha.maps = llm.L_lyman_alpha_to_map(halos_lyman_alpha,mapinst_lyman_alpha)

### Output map to file
llm.save_maps(mapinst)

### Calculate power spectrum
k,Pk,Nmodes = llm.map_to_pspec(mapinst,cosmo)
Pk_sampleerr = Pk/np.sqrt(Nmodes)

k_lyman_alpha,Pk_lyman_alpha,Nmodes_lyman_alpha = llm.map_to_pspec(mapinst_lyman_alpha,cosmo_lyman_alpha)
Pk_sampleerr_lyman_alpha = Pk_lyman_alpha/np.sqrt(Nmodes_lyman_alpha)


### Plot results
llm.plot_results(mapinst,k,Pk,Pk_sampleerr,params)

llm.plot_results(mapinst_lyman_alpha,k_lyman_alpha,Pk_lyman_alpha,Pk_sampleerr_lyman_alpha,params_lyman_alpha)

llm.write_time('Finished Line Intensity Mapper')
