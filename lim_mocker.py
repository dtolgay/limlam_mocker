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

import pprint

def L_x_to_mock_map(params, name_of_the_line):

	"""
	Computing and plotting the mock map

	Returns
	-------
	params : python file name  
	    Name of the python file name 
	name_of_the_line : string
	    Name of the line emission of interest
	"""

	llm.debug.verbose = True
	llm.write_time('Starting Line Intensity Mapper')

	### Setup maps to output
	mapinst   = llm.params_to_mapinst(params);

	### Load halos from catalogue
	halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file)
	halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst)

	### Calculate Luminosity of each halo
	halos.L    = llm.Mhalo_to_L(halos, params.model, params.coeffs)

	### Bin halo luminosities into map
	mapinst.maps = llm.L_to_map_2(halos,mapinst)   # The weights of the plot is now in terms of L_solar

	### Calculate power spectrum
	k,Pk,Nmodes = llm.map_to_pspec(mapinst,cosmo)
	Pk_sampleerr = Pk/np.sqrt(Nmodes)

	### Plot results
	llm.plot_results(mapinst,k,Pk,Pk_sampleerr,params)	

	llm.write_time('Finished Line Intensity Mapper for ' + name_of_the_line + "!")	

	return 0 

L_x_to_mock_map(params_lyman_alpha, 'Ly-α')
# L_x_to_mock_map(params, 'CO')
