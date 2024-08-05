######################################################################################################
######################################################################################################
###### In this file we take the raw particle output from the simulation hdf5 files and convert it into
###### a binary pickle file containing useful quantities to plot. Currently finds at each snapshot
# time
# galactocentric position of cluster CoM
# distance from cluster CoM to nearest DM particle

# galaxy mass enclosed by NFW scale radius
# galaxy circular velocity curves at start and end

### TO DO
# stream coordinates?
######################################################################################################
######################################################################################################

from amuse.lab import *
from amuse.ext.radial_profile import radial_profile, radial_density
import numpy as np
from amuse import io
import re
import pickle
import asyncio
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from amuse.plot import plot
sys.setrecursionlimit(10000)
fontprops = fm.FontProperties(size=12)

fname= 'sim_analytic_{:s}_df_model_{:s}_Mc{:1g}W{:g}X{:g}.hdf5'.format(str(True),str(False), 10000,5,
                                                                4.1)
plt.style.use('dark_background')
print('about to read cluster_' + fname)
data_cluster = io.read_set_from_file('cluster_'+fname, close_file=True)
print('about to read unbound_' + fname)
data_unbound = io.read_set_from_file('unbound_'+fname, close_file=True)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def plotting(cluster, unbound):
    t_snap = cluster.get_timestamp().in_(units.Myr)
    fig, ax = plt.subplots()
    # rotations so com velocity along x axis
    psi = np.pi-np.arctan2(cluster.copy().center_of_mass_velocity().y.value_in(units.km/units.s), cluster.copy().center_of_mass_velocity().x.value_in(units.km/units.s))
    if unbound:
        # compute the radial velocity dispersion of the unbound particles
        vels = unbound[unbound.position.lengths()<(6 | units.kpc)].copy().velocity.lengths()
        # unbound.position -= cluster.center_of_mass()
        # unbound.velocity -= cluster.center_of_mass_velocity()
        # unbound.position -= cluster.center_of_mass()
        unbound.rotate(0,0,psi)
        plt.scatter(unbound.x.value_in(units.kpc), unbound.y.value_in(units.kpc), s=1+2*unbound.mass.value_in(units.Msun), c='white', alpha=0.2)
    print(t_snap)
    print('number of bound particles', len(cluster))
    print('number of unbound particles', len(unbound))
    # cluster.move_to_center()
    print(cluster.center_of_mass_velocity())
    # cluster.position -= cluster.center_of_mass()
    cluster.rotate(0,0,psi)
    print(cluster.center_of_mass_velocity())
    plt.scatter(cluster.x.value_in(units.kpc), cluster.y.value_in(units.kpc), s=1+2*cluster.mass.value_in(units.Msun), c='r', alpha=0.1)
    # plt.scatter(cluster.center_of_mass().x.value_in(units.kpc), cluster.center_of_mass().y.value_in(units.kpc), s=20, c='r', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    h = 1
    plt.xlim(cluster.center_of_mass().x.value_in(units.kpc)-h, cluster.center_of_mass().x.value_in(units.kpc)+h)
    plt.ylim(cluster.center_of_mass().y.value_in(units.kpc)-h/2, cluster.center_of_mass().y.value_in(units.kpc)+h/2)
    # plt.text(-4.5, -5, 't = {:g} Myr'.format(t_snap.value_in(units.Myr)), fontsize=12)
    plt.text(cluster.center_of_mass().x.value_in(units.kpc)-h, cluster.center_of_mass().y.value_in(units.kpc)-h/2, 't = {:g} Myr'.format(t_snap.value_in(units.Myr)), fontsize=12, verticalalignment='bottom', horizontalalignment='left', color='white')
    plt.text(cluster.center_of_mass().x.value_in(units.kpc), cluster.center_of_mass().y.value_in(units.kpc)-h/2, 'To Galactic Centre', fontsize=12, verticalalignment='bottom', horizontalalignment='center', color='white')
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    plt.arrow(cluster.center_of_mass().x.value_in(units.kpc), cluster.center_of_mass().y.value_in(units.kpc)-h/4, 
              0, 
              -h/10, 
              head_width=0.05*h, head_length=0.05*h, fc='w', ec='w')
    scalebar = AnchoredSizeBar(ax.transData,
                           0.1*h, '{:g} kpc'.format(0.1*h), 'lower right', 
                           pad=0.0,
                           color='white',
                           frameon=False,
                           size_vertical=0.005*h,
                           fontproperties=fontprops)

    ax.add_artist(scalebar)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('zoomed_live_cluster/'+str(t_snap.value_in(units.Myr)).zfill(4)+'.png', dpi=500)
    i+=1

for cluster, unbound in zip(data_cluster.history, data_unbound.history):
    plotting(cluster, unbound)
    

    