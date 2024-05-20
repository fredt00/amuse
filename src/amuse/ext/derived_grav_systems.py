from amuse.units import constants
from amuse.datamodel import Particles, ParticlesSuperset
from amuse.datamodel.particle_attributes import HopContainer, particle_potential
from amuse.units import nbody_system
from amuse.ic.kingmodel import new_physical_king_model
from amuse.ic.brokenimf import new_masses
from amuse.couple import bridge
import numpy as np
from amuse.units.quantities import zero
from amuse.units import units
class center_of_mass(object):
    """
    com=center_of_mass(grav_instance)
    derived system, returns center of mass as skeleton grav system
    provides: get_gravity_at_point, get_potential_at_point
    """

    def __init__(self,baseclass):
        self.baseclass=baseclass

    def get_gravity_at_point(self,radius,x,y,z):
        mass=self.baseclass.total_mass
        xx,yy,zz=self.baseclass.get_center_of_mass_position()
        
        eps2=self.baseclass.parameters.epsilon_squared
        
        dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+eps2)
        
        ax=constants.G*mass*(xx-x)/dr2**1.5
        ay=constants.G*mass*(yy-y)/dr2**1.5
        az=constants.G*mass*(zz-z)/dr2**1.5
        
        return ax,ay,az

    def get_potential_at_point(self,radius,x,y,z):
        mass=self.baseclass.total_mass
        xx,yy,zz=self.baseclass.get_center_of_mass_position()
        
        eps2=self.baseclass.parameters.epsilon_squared
        dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+eps2)
        
        phi=-constants.G*mass/dr2**0.5
        
        return phi

class copycat(object):
    """
    copy=copycat(base_class,grav_instance, converter)
    derived system, returns copy of grav instance with
    get_gravity_at_point, get_potential_at_point reimplemented in 
    base_class
    """
    def __init__(self,baseclass, system,converter):
        self.baseclass=baseclass
        self.system=system
        self.converter=converter
          
    def get_gravity_at_point(self,radius,x,y,z):
        instance=self.baseclass(self.converter)

        instance.initialize_code()
        instance.parameters.epsilon_squared = self.system.parameters.epsilon_squared
        parts=self.system.particles.copy()
        instance.particles.add_particles(parts)

        ax,ay,az=instance.get_gravity_at_point(radius,x,y,z)
        
        instance.stop()
        return ax,ay,az

    def get_potential_at_point(self,radius,x,y,z):
        instance=self.baseclass(self.converter)

        instance.initialize_code()
        instance.parameters.epsilon_squared = self.system.parameters.epsilon_squared
        parts=self.system.particles.copy()
        instance.particles.add_particles(parts)

        phi=instance.get_potential_at_point(radius,x,y,z)
        
        instance.stop()
        return phi


# create a wrapper class for a gravity code to describe a star cluster including bound and unbound particles and stellar evolution
class star_cluster(object):
    """
    star_cluster=star_cluster(grav_instance,converter)
    derived system, returns star cluster system with
    get_gravity_at_point, get_potential_at_point reimplemented in 
    base_class
    """
    def __init__(self,code,code_converter, W0, r_tidal=None,r_half=None, n_particles=None, M_cluster=False, field_code=None,field_code_number_of_workers=1,code_number_of_workers=1):
        self.converter=code_converter
        self.code=code(self.converter, mode='openmp',number_of_workers=code_number_of_workers)
        self.field_code=field_code
        self.field_code_number_of_workers=field_code_number_of_workers

        self.r_tidal=r_tidal

        # create a scale free king model,then scale it to the desired mass and tidal/half mass radius scalign velocities accordingly
        self.initialize_king_model(n_particles, M_cluster, W0, r_tidal, r_half)

        # self.center_of_mass=center_of_mass(self.code.particles)
        self.unbound = Particles()

        # initialize the code for get_gravity_at_point and get_potential_at_point
        self.gravity_from_cluster = bridge.CalculateFieldForCodes(
            self.new_code_to_calculate_gravity,               
            input_codes=[self.code],                       
            )

    def new_code_to_calculate_gravity(self): 
            result = self.field_code(self.converter, number_of_workers=self.field_code_number_of_workers)  # this can be GPU based at some point
            return result
    # initialize the king model
    def initialize_king_model(self, n_particles, M_cluster, W0, r_tidal=None, r_half=None):
        # we either fix the number of stars, or the total mass (down to stochastic fluctuations)
        m_stars = new_masses(stellar_mass=M_cluster,number_of_stars=n_particles)
        cluster = new_physical_king_model(W0, masses=m_stars, tidal_radius=r_tidal, half_mass_radius=r_half)
        self.code.particles.add_particles(cluster)

    # get the gravity at a point
    def get_gravity_at_point(self,radius,x,y,z):
        # ax,ay,az=self.center_of_mass.get_gravity_at_point(radius,x,y,z) # here we should set radius automatically to the half mass radius (if it was plummer) - maybe diff for king?
        ax,ay,az=self.gravity_from_cluster.get_gravity_at_point(radius,x,y,z)
        return ax,ay,az
    
    # get the potential at a point
    def get_potential_at_point(self,radius,x,y,z):
        # phi=self.center_of_mass.get_potential_at_point(radius,x,y,z)
        phi = self.gravity_from_cluster.get_potential_at_point(radius,x,y,z)
        return phi
    
    # evolve the model to the specified time removing unbound particles and drifting them
    def evolve_model(self,tend):
        self.remove_unbound_particles_old()
        if len(self.unbound) > 0:
            self.drift_unbound_particles(tend-self.code.model_time)
        self.code.evolve_model(tend)
        
    # remove unbound particles from the system
    def remove_unbound_particles(self):
        core = self.bound.cluster_core(self.converter, density_weighting_power=2, reuse_hop=False, hop=HopContainer())
        position=self.bound.position-core.position
        velocity=self.bound.velocity-core.velocity
        v2=velocity.lengths_squared()
        r2=position.lengths_squared()
        # Compute potential of particles outside tidal radius
        boundary_radius2 = self.bound.LagrangianRadii(mf=[0.9])[0][0]**2#self.r_tidal**2

        outside_boundary = np.where((r2 > boundary_radius2))[0]
        outside_particles = self.bound[outside_boundary]
        if len(outside_particles)>0:
            # Compute potential of particles outside tidal radius
            pot = [] | units.m**2/units.s**2
            for particle in outside_particles:
                pot.append(particle_potential(self.bound,particle))
            print(pot)
            # Remove particles with total energy greater than zero
            unbound_indices = np.where((pot + 0.5 * v2[outside_boundary] > zero))[0]
            unbound_particles = outside_particles[unbound_indices]
            self.code.particles.remove_particles(unbound_particles)
            self.unbound.add_particles(unbound_particles)

    def remove_unbound_particles_old(self):
        bound = self.code.particles.bound_subset(unit_converter=self.converter,tidal_radius=self.bound.LagrangianRadii(mf=[0.9])[0][0], strict=True)
        new_unbound = self.code.particles.difference(bound)
        self.unbound.add_particles(new_unbound)
        self.code.particles.remove_particles(new_unbound)

    # drift unbound particles
    def drift_unbound_particles(self, dt):
       self.unbound.position += self.unbound.velocity * dt
    
    # the bound particles
    @property
    def bound(self):
        return self.code.particles
    
    # this should return all the particles so they are kicked correctly
    @property
    def particles(self):
        return ParticlesSuperset([self.bound, self.unbound])