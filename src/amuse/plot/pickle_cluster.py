######################################################################################################
######################################################################################################
###### In this file we take the raw particle output from the simulation hdf5 files and convert it into
###### a binary pickle file containing useful quantities to plot. Currently finds at each snapshot
# time
# galactocentric position of cluster CoM
# distance from cluster CoM to nearest DM particle



### TO DO
# stream coordinates?
# should be parallelised
######################################################################################################
######################################################################################################

from amuse.lab import *
import numpy
from amuse import io
import pickle
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.derived_grav_systems import tidal_field
import sys
from amuse.community.fastkick.interface import FastKick
from amuse.datamodel import ParticlesSuperset
import argparse
from amuse.datamodel.particle_attributes import HopContainer
from amuse.community.ph4.interface import ph4

sys.setrecursionlimit(10000)

# import petar

# def petar_cluster(filename):
#     particle = petar.Particle(interrupt_mode='bse', external_mode='galpy')
#     particle.loadtxt('../first_run/data.'+str(i),skiprows=1)
#     cluster = Particles(len(particle.pos))
#     cluster.position = particle.pos | units.pc
#     cluster.velocity = particle.vel | units.pc/units.Myr
#     cluster.mass = particle.mass | units.MSun
#     cluster.move_to_center()

# in case of petar files

def amuse_cluster(filename, galaxy_filename,data):
    print('about to read '+ filename)
    data_cluster = io.read_set_from_file(filename, close_file=True)
    data_galaxy = io.read_set_from_file(galaxy_filename, close_file=True)
    i=0
    unbound_particles = Particles()
    for cluster, galaxy in zip(data_cluster.history, data_galaxy.history):
        t_snap = cluster.get_timestamp().in_(units.Myr)
        print(t_snap.in_(units.Myr))

        gal_converter = nbody_system.nbody_to_si(galaxy.total_mass(), galaxy.total_radius())
        galaxy_force_field = FastKick(gal_converter, number_of_workers=20)
        galaxy_force_field.particles.add_particles(galaxy)
        galaxy_force_field.parameters.epsilon_squared=gal_converter.to_nbody((100 | units.pc)**2)
        gal_field = tidal_field(galaxy_force_field)
        # remove previous unbound particles
        cluster.remove_particles(unbound_particles)

        # compute who is bound and who isn't - note this is for old version where the code hasn't already done this for us
        converter= nbody_system.nbody_to_si(cluster.total_mass(), cluster.total_radius())
        # define a fastkick instance that we will use for all our potential calculations
        computer = ph4(converter, number_of_workers=23)
        computer.particles.add_particles(cluster)

        binaries= computer.particles.get_binaries(hardness=5)
        print("number of binaries", len(binaries))
        while True:
            # the particles in the framework that are currently defined as bound
            # find the centre of mass
            core = computer.particles.cluster_core(converter, density_weighting_power=2, reuse_hop=False, hop=HopContainer())
            position=computer.particles.position-core.position
            r2=position.lengths_squared()

            # find the particles outside the tidal radius - only compute energy of these
            tidal_radius = gal_field.tidal_radius(4|units.pc, core.position.x, core.position.y, core.position.z, computer.particles.total_mass())
           
            outside = computer.particles[r2 > tidal_radius**2]
            if len(outside) == 0:
                break

            energies = 0.5*(outside.velocity-core.velocity).lengths()**2 + outside.potential_in_code
            a_max = energies.argmax()
            # remove the particle with the highest positive energy - if all negative then break
            if energies[a_max] > 0 | units.erg/units.kg:
                # update the unbound particles
                to_remove = outside[a_max]
                unbound_particles.add_particle(to_remove)
                cluster.remove_particle(to_remove)
                computer.particles.remove_particle(to_remove)
            else:
                break

        galaxy_force_field.stop()
        print("number of bound particles", len(cluster))

        data['time'].append(t_snap)
        data['mass'].append(cluster.mass.sum())
        data['mean_mass'].append(cluster.mass.mean())
        data['galactocentric_radius'].append(cluster.center_of_mass().length())

        cluster.move_to_center()
        rhalf = cluster.LagrangianRadii(mf=[0.5])[0][0]
        data['rhalf'].append(rhalf)
        
        potential_energy = (computer.particles.mass*computer.particles.potential_in_code).sum()

        inside = cluster.position.lengths() < rhalf
        E = cluster.kinetic_energy() + potential_energy
        data["E"].append(E)
        data['psi'].append((cluster.mass[inside]**(5/2)).mean()/cluster.mass[inside].mean()**(5/2))
        data['kappa'].append(-E*rhalf/(constants.G*cluster.mass.sum()**2))
    return data

def model_cluster(filename, data):
    file = numpy.genfromtxt(filename)
    # note the data is stored as np.array([[self.model_time.value_in(units.Myr), x,y,z,vx,vy,vz, self.N,
    #  self.mbar.value_in(units.MSun), self.mbar_se.value_in(units.MSun), self.half_mass_radius.value_in(units.pc),
    # self.rtidal().value_in(units.pc), self.n_trhp, self.kappa, self.M_seg]])
    data['time'] = file[:,0] | units.Myr
    data['galactocentric_radius'] = numpy.sqrt(file[:,1]**2 + file[:,2]**2 + file[:,3]**2) | units.pc
    data['mass'] = file[:,7]*file[:,8] | units.MSun
    data['rhalf'] = file[:,10] | units.pc
    # data['psi'] = file[:,13] # within the half mass radius
    data['mean_mass'] = file[:,8] | units.MSun
    data['kappa'] = file[:,13]
    data['E'] = -data['kappa']*constants.G*data['mass']**2/data["rhalf"]
    data["RhJ"] = file[:,10]/file[:,11]
    return data 



def main(filename, cluster_file_type, galaxy_filename,outfile):
    print("reading in " + filename + " of type " + cluster_file_type + " and outputting to " + outfile)
    if galaxy_filename:
        print("using galaxy file " + galaxy_filename)
    data = {}
    data['time'] = [] | units.Myr
    data['galactocentric_radius'] = [] | units.kpc
    data['mass'] = [] | units.MSun
    data['rhalf'] = [] | units.pc
    data['E'] = [] | units.erg
    data['psi'] = [] # within the half mass radius
    data['mean_mass'] = [] | units.MSun
    data['kappa'] = []
    data["RhJ"] = [] | units.pc

    if cluster_file_type == 'hdf5':
        final_data = amuse_cluster(filename,galaxy_filename, data)
    elif cluster_file_type == 'txt':
        final_data = model_cluster(filename, data)
    else:
        print('File type not recognised')

    with open(outfile, 'wb') as handle:
        pickle.dump(final_data, handle)

    return 0

# The parser for taking the users inputs
def new_argument_parser():
    result = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # file specifications
    result.add_argument("--cluster_file", dest="filename", default = None,
                      help="file containing cluster data (default: %(default)s)")
    result.add_argument("--cluster_file_type", dest="cluster_file_type", default = 'hdf5', choices=['hdf5', "txt"],
                        help="Type of file to read in cluster simulationdata (default: %(default)s)")
    
    result.add_argument("--galaxy_file", dest="galaxy_filename", default = None,
                      help="file containing galaxy data - currently used for computing the tidal radius (default: %(default)s)")
    
    result.add_argument("--outfile", dest="outfile", default = None,
                      help="file for plotting data to be stored in (default: %(default)s)")
    
    return result

if __name__ == '__main__':
    arguments = new_argument_parser().parse_args()
    main(**arguments.__dict__) 