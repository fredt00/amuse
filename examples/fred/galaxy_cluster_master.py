import numpy as np
from amuse.lab import *
from amuse.community.petar.interface import petar
from amuse.community.fi.interface import Fi
from amuse.community.fastkick.interface import FastKick
from amuse.community.sse.interface import SSE
from amuse.couple import bridge
from amuse import io
from amuse.datamodel import Particles
import sys
from amuse.ext.dynamical_friction import dynamical_friction
from amuse.ext.derived_grav_systems import star_cluster
import amuse.ext.galactic_potentials as galactic_potentials
import inspect
import argparse
from amuse.ext.cluster_model import star_cluster_particle

def setup_live_galaxy(Nh=1e5, Mh=1e10 | units.MSun,Rscale=4.1 | units.kpc, t_settle=0|units.Myr, dt=1 | units.Myr, epsilon=88.6 | units.pc):
    converter= nbody_system.nbody_to_si(Mh, Rscale)
    # halo
    galaxy = new_halogen_model(Nh, converter, alpha=1, beta=3, gamma=1, 
                            scale_radius=Rscale, cutoff_radius=10.*Rscale)
    
    galaxy.move_to_center()
    # try fastkick for faster potential computation
    scaler = FastKick(converter, number_of_workers=20)
    scaler.epsilon_squared = converter.to_nbody(epsilon**2)
    scaler.particles.add_particles(galaxy)
    potential_energy = scaler.get_potential_energy()
    scaler.stop()
    galaxy.velocity*=(-2.*galaxy.kinetic_energy()/potential_energy)**-.5
    converter_gadget=nbody_system.nbody_to_si(dt, Mh)
    if t_settle>0|units.Myr:
        print('evolving galaxy IC to', t_settle.in_(units.Gyr), 'to allow it to settle')
        gravity_gal = Fi(converter_gadget,mode='openmp',redirection='file',redirect_file='output_fi.txt')
        gravity_gal.parameters.epsilon_squared=converter_gadget.to_nbody(epsilon**2)
        gravity_gal.parameters.use_hydro_flag=False
        gravity_gal.particles.add_particles(galaxy)
        channel_to_galaxy = gravity_gal.particles.new_channel_to(galaxy)
        gravity_gal.evolve_model(t_settle)
        channel_to_galaxy.copy()
        gravity_gal.stop()
        # recenter
        galaxy.move_to_center()
    return galaxy

def convert_inputs_to_galactic_potential(potential_option, potential_parameters, potential_units):
    num_parameters = len(potential_parameters)
    if num_parameters!=len(potential_units): 
        print('ERROR: you must specify units for all potential parameters')
        return -1
    input_args = len(inspect.getargs(getattr(galactic_potentials, potential_option).__init__.__code__).args)
    if input_args-2 != num_parameters and input_args>1:
        print('WARNING: you have not specified all potential parameters, default values will be used')
    print('setting up potential', potential_option, 'with parameters', potential_parameters)
    if num_parameters==0:
        return getattr(galactic_potentials, potential_option)()

    unit_converter = {'kpc': units.kpc, 'MSun/kpc3': units.MSun/units.kpc**3, 'MSun': units.MSun, 'None': units.none}
    for i in range(len(potential_parameters)):
        potential_parameters[i] = potential_parameters[i] | unit_converter[potential_units[i]]
    return getattr(galactic_potentials, potential_option)(*potential_parameters)

def read_hdf_and_get_requested_snapshot(filename, restart_time):
    all_snap = io.read_set_from_file(filename)
    for particles in all_snap.history:
        if len(particles)>1:
            if particles.get_timestamp() == restart_time:
                break
    if particles.get_timestamp() != restart_time:
        print('ERROR: requested restart time not found in file')
        return -1
    return particles

def configure_galaxy(N_halo, Mh, Rh, t_settle, galaxy_file, potential_option, potential_parameters, potential_units, analytic, restart_time,
                      dt,eps_gal_to_clu, galaxy_force_number_of_workers):
    if analytic:
        galaxy = convert_inputs_to_galactic_potential(potential_option, potential_parameters, potential_units)
        gravity_from_galaxy = None
    else:
        if galaxy_file:
            galaxy_particles = read_hdf_and_get_requested_snapshot(galaxy_file, restart_time)
        else:
            galaxy_particles = setup_live_galaxy(Nh=N_halo, Mh=Mh, Rscale=Rh,t_settle=t_settle, dt=dt, epsilon=eps_gal_to_clu)
        galaxy_converter = nbody_system.nbody_to_si(galaxy_particles.mass.sum(), dt)
 
        # set up code for evolution of galaxy - set OMP_NUM_THREADS to number of cores for this to use
        galaxy = Fi(galaxy_converter,mode='openmp',redirection='file',redirect_file='output_fi.txt')
        galaxy.parameters.epsilon_squared=galaxy_converter.to_nbody(eps_gal_to_clu**2)
        galaxy.parameters.use_hydro_flag=False
        galaxy.particles.add_particles(galaxy_particles)

        # in case we want to use a different softening from galaxy to cluster to the internal galaxy softening
        # set up direct sum gravity calculator for kicking cluster
        def new_galaxy_code_to_calculate_gravity():
            result = FastKick(galaxy_converter, number_of_workers=galaxy_force_number_of_workers)
            return result
        gravity_from_galaxy = bridge.CalculateFieldForCodes(
            new_galaxy_code_to_calculate_gravity,              
            input_codes=[galaxy],                       
            )
        
    return galaxy, gravity_from_galaxy

def setup_cluster_from_file(cluster_file, cluster_file_type, restart_time=0 | units.Myr):
    filename = cluster_file 
    print('reading in cluster IC from ' + filename)

    # hdf5 file in amuse format
    if cluster_file_type=='hdf5': cluster = read_hdf_and_get_requested_snapshot(filename, restart_time)
        
    # the mcluster option '-u 1' generates data in astronomical unit (Msun, pc, km/s)
    if cluster_file_type=="dat.10":
        data = np.genfromtxt(filename)
        cluster = Particles(len(data))
        cluster.mass = data[:,0] | units.MSun
        cluster.position = data[:,1:4] | units.pc
        cluster.velocity = data[:,4:7] | units.kms
    return cluster

def configure_cluster(N_cluster, M_cluster, W0, r_half, r_tidal, initial_position, initial_velocity, Vcirc_fraction, cluster_model,
                       cluster_file, cluster_file_type, restart_time, stellar_evolution, galaxy, analytic, dt,
                         star_cluster_number_of_workers):
    # default to solar
    if len(initial_position)==0: initial_position = [8,0,0] 
    if len(initial_velocity)==0: initial_velocity = [0,220,0]
    Rinit = initial_position | units.kpc
    Vinit = initial_velocity | units.kms
    
    if Vcirc_fraction:
        if analytic:
            Vcirc = galaxy.circular_velocity(Rinit.length())
        else:
            selection = (galaxy.particles.position).lengths()<Rinit.length()
            Menc=galaxy.particles[selection].mass.sum()
            Vcirc  =(constants.G * Menc/Rinit.length())**.5
        Vy = Vcirc_fraction * Vcirc
        Vinit = [0, Vy.value_in(units.kms), 0] | units.kms

    t_orb=(2 * np.pi*Rinit.length()/Vinit.length())
    print('initialising cluster on orbit with R=', Rinit, 'V=', Vinit, "t_orb=", t_orb.in_(units.Myr))
    print('bridge timestep/torb is', dt/t_orb)

    converter = nbody_system.nbody_to_si(M_cluster, dt)

    if cluster_model:
        cluster = star_cluster_particle(M_cluster, r_half, Rinit, Vinit, grav_instance=galaxy, stellar_evolution=stellar_evolution)
    else:
        cluster_particles = None
        if stellar_evolution: stellar_evolution=SSE
        if cluster_file:
            cluster_particles = setup_cluster_from_file(cluster_file, cluster_file_type, restart_time)
        cluster = star_cluster(code=petar, code_converter=converter, particles=cluster_particles, W0=W0, r_tidal=r_tidal,r_half=r_half, n_particles=N_cluster,
                                    M_cluster=M_cluster,code_number_of_workers=star_cluster_number_of_workers, stellar_evolution=stellar_evolution, field_code = galaxy, time=restart_time)
        if restart_time==0 | units.Myr:
            cluster.particles.position += Rinit
            cluster.particles.velocity += Vinit
    return cluster, Rinit, Vinit
    
# The main function that sets up the simulation and evolves it
def main(star_cluster_number_of_workers = 2, galaxy_force_number_of_workers = 0, N_halo = 10000, N_cluster = None, W0=5.0, r_half = None, r_tidal = None,
            M_cluster = None, t_end = 10 | units.Myr, restart_file=None, Mh=100|units.MSun, Rh=4.43 | units.kpc,
            output_interval=20 | units.Myr, t_settle = 1 | units.Gyr, initial_position = [], initial_velocity = [], Vcirc_fraction = None,
            eps_gal_to_clu = 100 | units.pc, dt=1.0|units.Myr, galaxy_file = None, cluster_model = False, cluster_file = None,
            cluster_file_type='hdf5', restart_time = 0 | units.Myr, df_model=False, analytic=False, 
            stellar_evolution=False, potential_option='MWpotentialBovy2015', potential_parameters = [], potential_units = []):
    # check input options
    print('Your specified options are', locals())

    # set the random seed - deprecated
    np.random.seed(123)
    if restart_file:
        cluster_file = "cluster_"+restart_file
        if not analytic:
            galaxy_file = "galaxy_"+restart_file
    # set up galaxy IC/potential
    galaxy, gravity_from_galaxy = configure_galaxy(N_halo, Mh, Rh, t_settle, galaxy_file, potential_option, potential_parameters,
                                                    potential_units, analytic, restart_time, dt,eps_gal_to_clu, galaxy_force_number_of_workers)
    
    # set up the cluster - new IC or read in
    cluster, Rinit, Vinit = configure_cluster(N_cluster, M_cluster, W0, r_half, r_tidal, initial_position, initial_velocity, Vcirc_fraction, cluster_model,
                       cluster_file, cluster_file_type, restart_time, stellar_evolution, galaxy, analytic, dt,
                         star_cluster_number_of_workers)

    if df_model:
        if cluster_model:
            dyn_fric = dynamical_friction(galaxy, cluster.particles, half_mass_radius = cluster.half_mass_radius) #
        else:
            dyn_fric = dynamical_friction(galaxy, cluster.bound.particles, half_mass_radius = cluster.half_mass_radius) # need rh to update!

    if not restart_file:
        restart_file= 'sim_analytic_{:s}_df_model_{:s}_Mc{:g}W{:g}R{:g}V{:g}.hdf5'.format(str(analytic),str(df_model),
                                                                                            M_cluster.value_in(units.MSun),W0,
                                                                                            Rinit.length().value_in(units.kpc), 
                                                                                            Vinit.length().value_in(units.kms))

        # store initial conditions
        io.write_set_to_file(cluster.particles,'cluster_'+restart_file,'hdf5', timestamp=restart_time, append_to_file=False)
        if not analytic:
            io.write_set_to_file(galaxy.particles,'galaxy_'+restart_file,'hdf5', timestamp=restart_time,append_to_file=False)

    # add them to bridge in correct configuration
    integrator=bridge.Bridge(verbose=True, timestep=dt, use_threading=True)
    integrator.time = restart_time

    if analytic:
        if df_model:
            integrator.add_system(cluster, (galaxy, dyn_fric,), do_sync=True)
        else:
            integrator.add_system(cluster, (galaxy,), do_sync=True)
        integrator.add_system(cluster.unbound, (galaxy, cluster,), do_sync=True)
    elif df_model:
        system=bridge.GravityCodeInField(cluster, (galaxy, df_model,), do_sync=True, verbose=True,
                    radius_is_eps = False, h_smooth_is_eps=False, zero_smoothing=False, softening_length_squared=eps_gal_to_clu**2)
        unbound_system=bridge.GravityCodeInField(cluster.unbound, (galaxy, cluster,), do_sync=True, verbose=True,
                    radius_is_eps = False, h_smooth_is_eps=False, zero_smoothing=False, softening_length_squared=eps_gal_to_clu**2)
        integrator.add_code(system)
        integrator.add_code(unbound_system)
        integrator.add_code(galaxy)
    else:
        # for now use the softening internal to fi code
        system=bridge.GravityCodeInField(cluster, (galaxy,), do_sync=True, verbose=True)#,
                    #radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False,softening_length_squared=eps_gal_to_clu**2)
        unbound_system=bridge.GravityCodeInField(cluster.unbound, (galaxy,cluster,), do_sync=True, verbose=True)#,
                    # radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False,softening_length_squared=eps_gal_to_clu**2)
        integrator.add_code(system)
        integrator.add_code(unbound_system)
        system_cluster=bridge.GravityCodeInField(galaxy, (cluster,), do_sync=True, verbose=True,
                    radius_is_eps=False, h_smooth_is_eps=False, zero_smoothing=False, softening_length_squared=(0.01 | units.pc)**2)
        integrator.add_code(system_cluster)

    sys.stdout.flush()
    # evolve the bridge to the requested time
    while integrator.time < t_end:
        integrator.evolve_model(integrator.time+dt)
        print('evolved to', integrator.time.in_(units.Myr)) 
        # save output
        if integrator.time.value_in(units.Myr) % output_interval.value_in(units.Myr)==0:
            # cluster.transfer_unbound_particles()
            if not analytic:
                print('cluster distance from galactic centre', (cluster.bound.particles.center_of_mass()- galaxy.particles.center_of_mass()).length().in_(units.kpc))
            else:
                print('cluster distance from galactic centre', cluster.bound.particles.center_of_mass().length().in_(units.kpc))
            io.write_set_to_file(cluster.particles,'cluster_'+restart_file,'hdf5', timestamp=integrator.time, append_to_file=True)
            if not analytic:
                io.write_set_to_file( galaxy.particles,'galaxy_'+restart_file,'hdf5', timestamp=integrator.time, append_to_file=True)
        sys.stdout.flush()

    # clean up
    if not analytic: galaxy.stop()
    cluster.stop()

# The parser for taking the users inputs
def new_argument_parser():
    result = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #### GLOBAL SIMULATION PARAMETERS 
    result.add_argument("-o","--output_interval", dest="output_interval", type=units.Myr, default = 20 | units.Myr,
                      help="time interval in Myr to save output to file (default: %(default)s)")
    result.add_argument("-f", "--restart_file", dest="restart_file", default = None,
                      help="restart file name (default: %(default)s)")
    result.add_argument("--restart_time", dest="restart_time", type=units.Myr, default = 0 | units.Myr,
                      help="time of restart file - only used if restart_file is provided (default: %(default)s)")
    result.add_argument("-t", "--end_time", dest="t_end", type= units.Myr,  default = 5000 | units.Myr,
                      help="end time of the simulation in Myr (default: %(default)s)")
    result.add_argument("--df_model", dest='df_model', action='store_true',
                      help="use semi-analytic model for dynamical friction? Analytic halo must have log_log_slope method. Turns off cluster->galaxy kick in the live galaxy case (default: %(default)s)")
    result.add_argument("--analytic", dest='analytic', action='store_true',
                      help="use analytic model halo and dynamical friction? (default: %(default)s)")
    result.add_argument("--stellar_evolution",  dest='stellar_evolution', action='store_true',
                      help="use stellar evolution in the cluster? (default: %(default)s)")
    result.add_argument("--star_cluster_number_of_workers", type=int, default = 2,
                        help="number of workers for star cluster code (note it uses openmp) (default: %(default)s)")
    result.add_argument("--galaxy_force_number_of_workers", type=int, default = 0,
                        help="number of workers for direct sum code calculating force from galaxy. if zero then uses Fi tree code (default: %(default)s)")
    
    ###### GALAXY OPTIONS
    result.add_argument("-N", "--halo_particle_number", dest="N_halo", type=int, default = 1e5,
                      help="number of stars in the galaxy dark matter halo (default: %(default)s)")
    result.add_argument("-M", "--halo_mass", dest="Mh", type=units.MSun, default = 1e10 | units.MSun,
                      help="galaxy halo mass (default: %(default)s)")
    result.add_argument("-R", '--halo_scale_radius', dest="Rh", type= units.kpc, default = 4.1 | units.kpc,
                      help="galaxy dark matter halo scale radius (default: %(default)s)")
    result.add_argument("-T","--Tsettle", dest="t_settle", type=units.Myr, default = 500 | units.Myr,
                      help="The time for which the galaxy initial condition is first simulated to allow it to relax (default: %(default)s)")
    result.add_argument("-g","--galaxy_file", dest="galaxy_file", default = None,
                      help="A file to read in Nbody initial condition for the galaxy (default: %(default)s)")
    
    # in case of analytic
    result.add_argument("--potential_option", dest='potential_option', choices= [x for x in dir(galactic_potentials) if inspect.isclass(getattr(galactic_potentials, x))][2:], 
                        default='MWpotentialBovy2015',
                      help="choice of potential profile for the galaxy halo, options in amuse/ext/galactic_potentials.py (default: %(default)s)"),
    # analytic inputs to be given in order - units to be given in next argument in same order!
    result.add_argument("--potential_parameters", dest="potential_parameters", action="append", default = [],type=float,
                        help="parameters for the potential profile in the order they appear in the class defintions (default: %(default)s)")
    result.add_argument("--potential_units", dest="potential_units", action="append", choices=['kpc','MSun/kpc3','MSun','None'],
                         default = [], type=str,
                        help="units for the parameters for the potential profile in the order they appear in the class defintions (default: %(default)s)")
    
    ######## CLUSTER OPTIONS
    # use subgrid cluster EMACSS?
    result.add_argument("--cluster_model", dest='cluster_model', action='store_true',
                        help="use subgrid cluster model from EMACSS + shocks? (default: %(default)s)")

    # pre defined IC
    result.add_argument("--cluster_file", dest="cluster_file", default = None,
                      help="A file to read in initial cluster condition (default: %(default)s)")
    result.add_argument("--cluster_file_type", dest="cluster_file_type", default = 'hdf5', choices=['hdf5', 'txt', 'dat.10'],
                      help="Type of file to read in cluster condition (default: %(default)s)")
    
    result.add_argument("-W", dest="W0", type=float, default = 5.0, # 5 is typical of open clusters and rapidly dissolving GCs, 7 for older, core collapsed objects
                      help="Dimension-less depth of the King potential (W0) (default: %(default)s)")
    result.add_argument("-n", dest="N_cluster", type=int, default = None,
                      help="number of stars in the cluster (default: %(default)s)") # note that currently we have equal mass stars so no option needed for that yet
    result.add_argument("--r_half", dest="r_half", type=units.parsec, default = 4.35|units.parsec,
                      help="cluser half mass radius (default: %(default)s)")
    result.add_argument("--r_tidal", dest="r_tidal", type=units.parsec, default = None |units.parsec,
                      help="cluser tidal radius (default: %(default)s)")
    result.add_argument("--M_cluster",  dest="M_cluster", type=units.MSun, default = 1e4 | units.MSun,
                      help="mass of the cluster (default: %(default)s)")
    
    result.add_argument("-X", "--initial_position", dest="initial_velocity", type=float, default = [],action="append",
                      help="cluser galactocentric initial position in kpc - specify 3 times for x,y,z. If empty, solar used (default: %(default)s)")     
    result.add_argument("-V", "--initial_velocity", dest="initial_velocity", type=float, default = [], action="append",  
                        help="cluser initial velocity in kms - specify 3 times for x,y,z. If empty, solar used (default: %(default)s)")  
    
    result.add_argument("--Vcirc_fraction", dest="Vcirc_fraction", type=float, default =None,
                      help="Fraction of circular velocity for initial cluster velocity - overides initial_velocity (default: %(default)s)")  
    
    result.add_argument("-e", "--epsilon", dest="eps_gal_to_clu", type=units.parsec, default = 100 |units.pc,
                      help="softening length used for velocity kicks from galaxy to cluster (default: %(default)s)") 
    
    return result

if __name__ == '__main__':
    arguments = new_argument_parser().parse_args()
    main(**arguments.__dict__)