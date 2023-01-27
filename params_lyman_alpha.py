# This is the parameter file for all halo
# and map parameters

### L_lyman_alpha(M, z, ...) model
model   = 'Santos' # Name of the model
coeffs  = None # specify None for default coeffs

### Halo parameters
halo_catalogue_file = 'catalogues/peakpatchcatalogue_1pt4deg_z2pt4-3pt4.npz'
min_mass            = 2.5e10

### Map parameters
nu_rest = 2470000 	# rest frame frequency of CO(1-0) transition in GHz
nu_i    = 728550.36    	# GHz
nu_f    = 557126.7459

nmaps   = 100
fov_x   = 1.4    # in degrees
fov_y   = 1.4    # in degrees
npix_x  = 256
npix_y  = 256 

map_output_file = './L_lyman_alpha_cube_trial'

### Plot parameters 
plot_cube = True
plot_pspec = True
