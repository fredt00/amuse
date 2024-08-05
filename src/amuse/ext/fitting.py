######### routines for fitting particle sets in AMUSE
import numpy as np

def bin_particles_density(radii, masses, num_bins):
    bin_edges = np.logspace(np.log10(min(radii)), np.log10(max(radii)), num_bins + 1)
    binned_density = np.histogram(radii, bins=bin_edges, weights=masses)[0]/(4/3 *np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3))
    bin_centers = (bin_edges[:-1]*bin_edges[1:])**0.5
    return bin_centers, binned_density

def nfw_density(radius, scale_radius, rho_0):
    x = radius / scale_radius
    return rho_0 / (x * (1 + x)**2)

def mass_enclosed(radius, scale_radius, rho_0):
    return 4*np.pi*rho_0*scale_radius**3*(np.log(1 + radius/scale_radius) - radius/(scale_radius+radius))

# Define the fitting function
def fit_function_menc(parameters, radius, density):
    scale_radius, rho_0 = parameters
    predicted_density = mass_enclosed(radius, scale_radius, rho_0)
    return np.sum((predicted_density - density)**2)


#BAD CODE!!
def setup_analytic_halo(galaxy):
    galaxy.move_to_center()
    galaxy = galaxy.select(lambda r : 100 | units.pc<r.length()<41 | units.kpc, ['position'])
    bin_centers, binned_density = bin_particles_density(galaxy.position.lengths().value_in(units.kpc),
                                                        galaxy.mass.value_in(units.MSun), 50)
    # Perform the fitting
    initial_guess = [4.43, 10**6]  # Initial guess for scale radius and rho_0
    # try mass enclosed
    menc = []
    radii = galaxy.position.lengths()
    for radius in bin_centers:
        selection = galaxy[radii<radius | units.kpc]
        if selection.mass.sum()>0 | units.MSun:
            menc.append(selection.mass.sum().value_in(units.MSun))
    options = {'maxiter': 1000} 
    result_menc = minimize(fit_function_menc, initial_guess, args=(bin_centers, menc), options=options)
    scale_radius_fit_menc=result_menc.x[0] | units.kpc
    rho_0_fit_menc = result_menc.x[1] | units.MSun/units.kpc**3
    fit_cost=result_menc.fun
    print('result from fitting:')
    print('fit cost=', fit_cost)
    print('rs=', scale_radius_fit_menc.in_(units.kpc))
    print('rho0=', rho_0_fit_menc.in_(units.MSun/units.kpc**3))
    halo_model = galactic_potentials.NFW_profile(rho_0_fit_menc, scale_radius_fit_menc)
    return halo_model