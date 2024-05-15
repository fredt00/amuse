from optparse import OptionParser
from amuse.units.optparse import OptionParser
import numpy as np
from amuse.lab import *
from amuse.community.petar.interface import petar
from amuse.community.fastkick.interface import FastKick
from amuse.couple import bridge
from amuse import io
from amuse.datamodel import Particles, ParticlesSuperset
import sys
import os
from scipy.optimize import minimize
import math
from amuse.ext.dynamical_friction import dynamical_friction, NFW_profile

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

## up here we should define as many functions as possible for setting up ICs, reading from files, and evolving for use in main
def setup_cluster(n,mstar,Rcluster,W0,Rinit,Vinit):
    Mcluster = n * mstar
    converter= nbody_system.nbody_to_si(Mcluster,Rcluster)
    np.random.seed(123)
    stars = new_king_model(int(n), W0,convert_nbody=converter)
    stars.mass=mstar
    stars.move_to_center()
    stars.position+= Rinit
    stars.velocity+= Vinit
    stars.radius = 1 | units.RSun
    return stars,converter

def setup_test_particle(Rinit,Vinit, mass):
    converter= nbody_system.nbody_to_si(mass,Rinit[0])
    np.random.seed(123)
    particle = Particles(1)
    particle.mass=mass
    particle.position= Rinit
    particle.velocity= Vinit
    # particle.radius = 4.35 | units.pc
    return particle,converter

def setup_galaxy(Nh=1e5, Mh=1e10 | units.MSun,Rscale=4.1 | units.kpc, t_settle=0|units.Myr, gadget_options={}, beta=3, dt=1 | units.Myr, do_scale=False):
    converter= nbody_system.nbody_to_si(Mh, Rscale)
    np.random.seed(123)
    galaxy = new_halogen_model(Nh, converter, alpha=1, beta=beta, gamma=1, 
                            scale_radius=Rscale,cutoff_radius=10.*Rscale)
    
    galaxy.move_to_center()
    if do_scale:
        # try fastkick for faster potential computation
        scaler = FastKick(converter, number_of_workers=20)
        scaler.epsilon_squared = converter.to_nbody(gadget_options['epsilon_squared'])
        scaler.particles.add_particles(galaxy)
        potential_energy = scaler.get_potential_energy()
        scaler.stop()
        galaxy.velocity*=(-2.*galaxy.kinetic_energy()/potential_energy)**-.5
    converter_gadget=nbody_system.nbody_to_si(dt, Mh)
    if t_settle>0|units.Myr:
        print('evolving galaxy IC to', t_settle.in_(units.Gyr), 'to allow it to settle')

        gravity_gal = Fi(converter_gadget,mode='openmp',redirection='file',redirect_file='output_fi.txt')
        gravity_gal.parameters.epsilon_squared=converter_gadget.to_nbody(gadget_options['epsilon_squared'])
        gravity_gal.parameters.use_hydro_flag=False
        gravity_gal.particles.add_particles(galaxy)
        channel_to_galaxy = gravity_gal.particles.new_channel_to(galaxy)
        gravity_gal.evolve_model(t_settle)
        channel_to_galaxy.copy()
        gravity_gal.stop()
    # recenter
    galaxy.move_to_center()
    return galaxy,converter_gadget

def setup_analytic_halo(galaxy):
    galaxy.move_to_center()
    galaxy = galaxy.select(lambda r : 100 | units.pc<r.length()<43 | units.kpc, ['position'])
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
    halo_model = NFW_profile(rho_0_fit_menc, scale_radius_fit_menc)
    return halo_model


# The main function that sets up the simulation and evolves it
def main(Nh=10000, n=100, W0=5.0, t_end=10|units.Myr,restart_file=None, Mh=100|units.MSun,
          Rh=4.1 | units.kpc,Rvir=1|units.parsec,diagnostic=20, t_settle=1|units.Gyr,
            Xinit=4.43 | units.kpc, eps_gal_to_clu = 100 | units.pc, dt=1.0|units.Myr, galaxy_file = None, beta=3,
            mstar= 1 |units.MSun, do_scale=False, df_model=False, analytic=False, r_half=4.35|units.parsec):
    options = locals()
    print('your specified options are', options)
    if do_scale== 'False': do_scale=False
    if do_scale== 'True': do_scale=True
    if df_model== 'False': df_model=False
    if df_model== 'True': df_model=True
    if analytic== 'False': analytic=False
    if analytic== 'True': analytic=True
    gadget_options = {'number_of_workers' : 27, 'epsilon_squared' : (88.6  | units.pc)**2, 'begin_time': 0.0 | units.Myr,
                       'max_size_timestep':2*dt,'time_max':dt*2.**14., 'time_limit_cpu': 0.1 | units.yr,
                       'timestep_accuracy_parameter':0.01, 'opening_angle':0.5}

    if restart_file:
        # if restart then read in the file - we can probably just see if file is set and so don't need other restart option thing
        all_snap_cluster = io.read_set_from_file('cluster_'+restart_file)
        all_snap_galaxy = io.read_set_from_file('galaxy_'+restart_file)
        # find a way to get the last snapshot without looping through
        for snapshot in all_snap_galaxy.history:
            restart_time = snapshot.get_timestamp().in_(units.Myr)
        galaxy = snapshot
        for snapshot in all_snap_cluster.history:
            if snapshot.get_timestamp().in_(units.Myr)==restart_time: break
        cluster = snapshot
        converter_gal = nbody_system.nbody_to_si(galaxy.mass.sum(),dt)
        converter_clu = nbody_system.nbody_to_si(cluster.mass.sum(),dt)
        print('restarting from', restart_time, ' using file', restart_file)
        del(all_snap_galaxy)
        del(all_snap_cluster)
        del(snapshot)
    else:
         # define the file name and save ICs. The file name should be some combination of parameters
        restart_file= 'sim_analytic_{:s}_df_model_{:s}_n{:1g}m{:g}W{:g}X{:g}.hdf5'.format(str(analytic),str(df_model), n,mstar.value_in(units.MSun),W0,
                                                                Xinit.value_in(units.kpc))
        print('setting up ICs to be saved to ' + restart_file)
        if os.path.exists('cluster_'+restart_file):
            print('ERROR: the output file already exists!') 
            return -1
        restart_time = 0 |units.Myr
        # set up the galaxy
        if galaxy_file:
            print('reading in galaxy IC from ' + galaxy_file)
            galaxy_all = io.read_set_from_file(galaxy_file, close_file=True)
            for snapshot in galaxy_all.history:
                galaxy= snapshot.select(lambda m: m<9.99e4 | units.MSun,["mass"])
                break
    # bin the particles
            galaxy.move_to_center()
            converter_gal = nbody_system.nbody_to_si(galaxy.mass.sum(),dt)
            del(galaxy_all)
        else:
            galaxy, converter_gal, = setup_galaxy(Nh=Nh, Mh=Mh, Rscale=Rh,t_settle=t_settle,gadget_options=gadget_options, beta=beta, dt=dt, do_scale=do_scale)
            
        # set up the cluster
        Rinit = [Xinit.value_in(units.kpc), 0, 0] | units.kpc

        # set up the semi-analytic dynamical friction model
        if df_model or analytic:
            halo_model = setup_analytic_halo(galaxy)
        
        if analytic:
            Vy = halo_model.circular_velocity(Xinit)
        else:
            # instead calculate based on the mass enclosed within the radius
            selection = (galaxy.position).lengths()<Rinit[0]
            Menc=galaxy[selection].mass.sum()
            Vy  =(constants.G * Menc/Rinit.length())**.5

        Vinit = [0,Vy.value_in(units.kms), 0] | units.kms
        print('initialising cluster on orbit with R=', Rinit, 'V=', Vinit)
        t_orb=(2 * np.pi*Xinit/Vy)
        print('bridge timestep/torb is', dt/t_orb)
        if n==1:
            cluster, converter_clu = setup_test_particle(Rinit, Vinit, mstar)
        else:
            cluster, converter_clu = setup_cluster(n,mstar,Rvir, W0,Rinit,Vinit)
        io.write_set_to_file(cluster,'cluster_'+restart_file,'hdf5', timestamp=restart_time,append_to_file=False)
        if not analytic:
            io.write_set_to_file(galaxy,'galaxy_'+restart_file,'hdf5', timestamp=restart_time,append_to_file=False)

    if not analytic:
        # set up code for evolution of galaxy
        gravity_gal = Fi(converter_gal,mode='openmp',redirection='file',redirect_file='output_fi.txt')
        gravity_gal.parameters.epsilon_squared=converter_gal.to_nbody(gadget_options['epsilon_squared'])
        gravity_gal.parameters.use_hydro_flag=False
        gravity_gal.particles.add_particles(galaxy)
        channel_to_galaxy = gravity_gal.particles.new_channel_to(galaxy)

    # set up code for evolution of cluster
    converter_petar = nbody_system.nbody_to_si(dt, cluster.total_mass())
    # test mass or cluster
    if n==1:
        gravity_clu = BHTree(converter_petar, number_of_workers = 1)
        gravity_clu.parameters.timestep = 1/2. | nbody_system.time
    else:
        gravity_clu = petar(converter_petar, mode='openmp', number_of_workers = 2)
    gravity_clu.particles.add_particles(cluster)
    channel_to_cluster = gravity_clu.particles.new_channel_to(cluster)

    if df_model or analytic:
        df_model = dynamical_friction(halo_model, gravity_clu, r_half = (.5**(-2/3)-1)**-.5 * (10 | units.pc) )

    if not analytic:
        # set up gravity calculators
        def new_galaxy_code_to_calculate_gravity():
            result = FastKick(converter_gal, number_of_workers=13)
            # result.epsilon_squared=converter_gal.to_nbody(eps_gal_to_clu**2) ####ERROR HERE eps is taken from source code not here!
            return result
        gravity_from_galaxy = bridge.CalculateFieldForCodes(
            new_galaxy_code_to_calculate_gravity,              # the code that calculates the acceleration field
            input_codes=[gravity_gal],                       # the codes to calculate the acceleration field of
            )
        
        def new_code_to_calculate_gravity(): 
            result = FastKick(converter_petar, number_of_workers=13)  # this can be GPU based at some point
            return result
        gravity_from_cluster = bridge.CalculateFieldForCodes(
            new_code_to_calculate_gravity,               
            input_codes=[gravity_clu],                       
            )
    
    # set up the bridge
    print('bridge time step is', dt.in_(units.myr))
    
    integrator=bridge.Bridge(verbose=True,timestep=dt,use_threading=True)
    
    if analytic:
        integrator.add_system(gravity_clu,(df_model,halo_model,), do_sync=True)
    elif df_model:
        system=bridge.GravityCodeInField(gravity_clu, (gravity_from_galaxy,df_model,), do_sync=True, verbose=True,
                    radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False,softening_length_squared=eps_gal_to_clu**2)
        integrator.add_code(system)
        integrator.add_code(gravity_gal)
    else:
        system=bridge.GravityCodeInField(gravity_clu, (gravity_from_galaxy,), do_sync=True, verbose=True,
                    radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False,softening_length_squared=eps_gal_to_clu**2)
        integrator.add_code(system)

        system_cluster=bridge.GravityCodeInField(gravity_gal, (gravity_from_cluster,), do_sync=True, verbose=True,
                    radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False,softening_length_squared=(0.01 | units.pc)**2)
        integrator.add_code(system_cluster)

    # evolve the bridge to the requested time
    time = 0 | units.myr
    while time < t_end:
        time +=dt
        integrator.evolve_model(time)
        print('evolved to', time.in_(units.Myr)) 
        channel_to_cluster.copy_attributes(["mass", 'x','y','z', 'vx', 'vy', 'vz'])
        if not analytic:
            channel_to_galaxy.copy_attributes(["mass", 'x','y','z', 'vx', 'vy', 'vz'])
            print('cluster distance from galactic centre', (cluster.center_of_mass()-galaxy.center_of_mass()).length().in_(units.kpc))
        else:
            print('cluster distance from galactic centre', cluster.center_of_mass().length().in_(units.kpc))
        # save output
        if time.value_in(units.Myr) % diagnostic.value_in(units.Myr)==0:
            io.write_set_to_file(cluster,'cluster_'+restart_file,'hdf5', timestamp=restart_time+time,append_to_file=True)
        sys.stdout.flush()
    gravity_gal.stop()
    gravity_clu.stop()

# The parser for taking the users inputs following the python script at input
def new_option_parser():
    result = OptionParser()
    #### GLOBAL SIMULATION PARAMETERS 
    result.add_option("-d", unit=units.Myr,dest="diagnostic", type="float", default = 20|units.Myr,
                      help="time interval in Myr to save output to file [%default]")
    result.add_option("-f", dest="restart_file", default = None,
                      help="restart file name [%default]")
    result.add_option("-t", unit=units.Myr,
                      dest="t_end", type="float", default = 5|units.Gyr,
                      help="end time of the simulation [%default]")
    result.add_option("--do_scale", 
                      type='choice', choices=('True', 'False'), dest='do_scale', default='True',
                      help="scale galaxy velocities according to softening length? [%default]")
    result.add_option("--df_model", 
                      type='choice', choices=('True', 'False'), dest='df_model', default='False',
                      help="use semi-analytic model for dynamical friction instead of satellite kicking galaxy? [%default]")
    result.add_option("--analytic", 
                      type='choice', choices=('True', 'False'), dest='analytic', default='False',
                      help="use analytic model halo and dynamical friction? [%default]")
    
    ###### GALAXY OPTIONS
    result.add_option("-N", "--Nhalo", dest="Nh", type="int",default = 1e6,
                      help="number of stars in the galaxy dark matter halo [%default]")
    result.add_option("-M", unit=units.MSun,
                      dest="Mh", type="float",default = 1e10|units.MSun,
                      help="galaxy halo mass [%default]")
    result.add_option("-R",  unit=units.kpc,
                      dest="Rh", type="float",default = 4.43|units.kpc,
                      help="galaxy halo scale radius [%default]")
    result.add_option("-T","--Tsettle", unit=units.Myr,
                      dest="t_settle", type="float", default = 2|units.Gyr,
                      help="The time for which the galaxy initial condition is first simulated with gadget2 params to allow it to relax [%default]")
    result.add_option("-g","--galfile",
                      dest="galaxy_file", default = None,
                      help="A file to read in initial [%default]")
    
    ######## CLUSTER OPTIONS
    result.add_option("-W", dest="W0", type="float", default = 5.0, # 5 is typical of open clusters and rapidly dissolving GCs, 7 for older, core collapsed objects
                      help="Dimension-less depth of the King potential (W0) [%default]")
    result.add_option("-n", dest="n", type="int",default = 1e4,
                      help="number of stars in the cluster [%default]") # note that currently we have equal mass stars so no option needed for that yet
    result.add_option("-r", "--Rcluster", unit=units.parsec,
                      dest="Rvir", type="float",default = 3.0|units.parsec,
                      help="cluser virial radius [%default]")
    result.add_option("--r_half", unit=units.parsec,
                      dest="r_half", type="float",default = 4.35|units.parsec,
                      help="cluser virial radius [%default]")
    result.add_option("-m", unit=units.MSun,
                      dest="mstar", type="float",default = 1 |units.MSun,
                      help="mass of a star in the cluster [%default]")
    result.add_option("-X", "--Rx", unit=units.kpc,
                      dest="Xinit", type="float",default = 4.43|units.kpc,
                      help="cluser galactocentric radius [%default]")           # for now we will just set it off at true circular velocity
    result.add_option("-e", "--epsilon", unit=units.parsec,
                      dest="eps_gal_to_clu", type="float",default = 88.6 |units.pc,
                      help="softening length used for velocity kicks from galaxy to cluster  [%default]") 

    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    main(**o.__dict__)