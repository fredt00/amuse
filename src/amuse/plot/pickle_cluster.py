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

def amuse_cluster(filename, data):
    print('about to read '+ filename)
    data_cluster = io.read_set_from_file('cluster_'+filename, close_file=True)
    for cluster in data_cluster.history:
        t_snap = cluster.get_timestamp().in_(units.Myr)
        print(t_snap.in_(units.Myr))

        converter= nbody_system.nbody_to_si(cluster.total_mass(), cluster.total_radius())
        bound = cluster.bound_subset(tidal_radius=80 | units.pc, unit_converter=converter).copy()
        # unbound = cluster.difference(bound).copy()

        print("number of bound particles", len(bound))

        data['time'].append(t_snap)
        data['mass'].append(bound.mass.sum())
        data['mean_mass'].append(bound.mass.mean())
        data['galactocentric_radius'].append(bound.center_of_mass().length())

        bound.move_to_center()
        rhalf = bound.LagrangianRadii(mf=[0.5])[0][0]
        data['rhalf'].append(rhalf)
        
        # this could happen in parallel - would need threadfence at end
        converter= nbody_system.nbody_to_si(bound.total_mass(), rhalf[0])
        scaler = FastKick(converter, number_of_workers=20)
        scaler.particles.add_particles(bound)
        potential_energy = scaler.get_potential_energy()
        scaler.stop()

        inside = bound.position.lengths() < rhalf
        
        data['psi'].append((bound.mass[inside]**(5/2)).mean()/bound.mass[inside].mean()**(5/2))
        data['kappa'].append(-(bound.kinetic_energy() + potential_energy)*rhalf/(constants.G*bound.mass.sum()**2))
    return data

def model_cluster(filename, data):
    file = numpy.genfromtxt(filename)
    # note the data is stored as np.array([[self.model_time.value_in(units.Myr), x,y,z,vx,vy,vz, self.N,
    #  self.mbar.value_in(units.MSun), self.mbar_se.value_in(units.MSun), self.half_mass_radius.value_in(units.pc),
    # self.rtidal().value_in(units.pc), self.n_trhp, self.kappa, self.M_seg]])
    data['time'] = file[:,0] | units.Myr
    data['galactocentric_radius'] = numpy.sqrt(file[:,1]**2 + file[:,2]**2 + file[:,3]**2) | units.kpc
    data['mass'] = file[:,7]*file[:,8] | units.MSun
    data['rhalf'] = file[:,10] | units.pc
    # data['psi'] = file[:,13] # within the half mass radius
    data['mean_mass'] = file[:,8] | units.MSun
    data['kappa'] = file[:,13]
    data['E'] = -data['kappa']*constants.G*data['mass']**2/data["rhalf"]
    data["RhJ"] = file[:,10]/file[:,11]
    return data 



def main(filename, cluster_file_type, outfile):
    print("reading in " + filename + " of type " + cluster_file_type + " and outputting to " + outfile)
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
        final_data = amuse_cluster(filename, data)
    elif cluster_file_type == 'txt':
        final_data = model_cluster(filename, data)
    else:
        print('File type not recognised')

    with open(outfile+'.pickle', 'wb') as handle:
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
    
    result.add_argument("--outfile", dest="outfile", default = None,
                      help="file for plotting data to be stored in (default: %(default)s)")
    
    return result

if __name__ == '__main__':
    arguments = new_argument_parser().parse_args()
    main(**arguments.__dict__) 