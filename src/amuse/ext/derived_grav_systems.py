from amuse.units import constants
from amuse.datamodel import Particles, ParticlesSuperset
from amuse.units import nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.ic.brokenimf import new_kroupa_mass_distribution
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
    def __init__(self,code,code_converter, r_half, W0, r_vir, r_tidal, n_particles, M_cluster):
        self.converter=code_converter
        self.code=code(self.converter)

        # create a scale free king model,then scale it to the desired mass and tidal/half mass radius scalign velocities accordingly
        self.initialize_king_model(n_particles, W0, r_tidal, r_half, M_cluster)

        self.center_of_mass=center_of_mass(self.code.particles)
        self.unbound= Particles()

    # initialize the king model
    def initialize_king_model(self, n_particles, W0, r_tidal, M_cluster):
        temp, rt_rvir = new_king_model(1, W0, return_rt_rvir=True)
        # we either fix the number of stars, or the total mass (down to stochastic fluctuations)
        m_stars = new_kroupa_mass_distribution(n_particles)
        converter = nbody_system.nbody_to_si(M_cluster, r_tidal/rt_rvir)
        cluster = new_king_model(n_particles, W0, do_scale=True, convert_nbody=converter)
        self.code.particles.add_particles(cluster)

    # get the gravity at a point
    def get_gravity_at_point(self,radius,x,y,z):
        ax,ay,az=self.center_of_mass.get_gravity_at_point(radius,x,y,z)
        return ax,ay,az
    
    # evolve the model to the specified time removing unbound particles and drifting them
    def evolve_model(self,tend):
        self.remove_unbound_particles()
        self.drift_unbound_particles(tend-self.code.model_time)
        self.code.evolve_model(tend)
        
    # remove unbound particles from the system
    def remove_unbound_particles(self):
        bound=self.code.particles.bound_subset(self.converter)
        unbound = self.code.particles.difference(bound)
        self.unbound.add_particles(unbound)
        self.code.particles.remove_particles(unbound)
    
    # drift unbound particles
    def drift_unbound_particles(self, dt):
       self.unbound.position += self.unbound.velocity * dt

    # get the potential at a point
    def get_potential_at_point(self,radius,x,y,z):
        phi=self.center_of_mass.get_potential_at_point(radius,x,y,z)
        return phi
    
    # the bound particles
    @property
    def bound(self):
        return self.code.particles
    
    # the unbound particles
    @property
    def unbound(self):
        return self.unbound
    
    # this should return all the particles so they are kicked correctly
    @property
    def particles(self):
        return ParticlesSuperset([self.bound, self.unbound])