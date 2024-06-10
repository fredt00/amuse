from amuse.units import constants
from amuse.datamodel import Particles, ParticlesSuperset
from amuse.ic.kingmodel import new_physical_king_model
from amuse.ic.brokenimf import new_masses
from amuse.couple import bridge
from amuse.units.quantities import zero
from amuse.units import units
import math
from amuse.units import nbody_system
class center_of_mass(object):
    """
    com=center_of_mass(grav_instance)
    derived system, returns center of mass as skeleton grav system
    provides: get_gravity_at_point, get_potential_at_point
    """

    def __init__(self,baseclass):
        self.baseclass=baseclass

    def get_gravity_at_point(self,radius,x,y,z):
        mass=self.baseclass.total_mass()
        xx,yy,zz=self.baseclass.center_of_mass()
        
        # eps2=self.baseclass.parameters.epsilon_squared+radius**2
        
        # dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+eps2)
        dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+radius**2)
        
        ax=constants.G*mass*(xx-x)/dr2**1.5
        ay=constants.G*mass*(yy-y)/dr2**1.5
        az=constants.G*mass*(zz-z)/dr2**1.5
        
        return ax,ay,az

    def get_potential_at_point(self,radius,x,y,z):
        mass=self.baseclass.total_mass()
        xx,yy,zz=self.baseclass.center_of_mass()
        
        eps2=self.baseclass.parameters.epsilon_squared + radius**2
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
    def __init__(self,code,code_converter,bound_particles=None ,unbound_particles=None,W0=5, r_tidal=None | units.pc,r_half=None | units.pc, n_particles=None,
                  M_cluster=False, field_code=None,field_code_number_of_workers=1,code_number_of_workers=1, field_code_mode = 'direct', stellar_evolution = None):
        self.converter=code_converter
        self.bound=code(self.converter, mode='openmp',number_of_workers=code_number_of_workers)
        self.unbound = drifter(stellar_evolution=stellar_evolution)
        if bound_particles:
            self.bound.particles.add_particles(bound_particles)
        else:
        # create a scale free king model,then scale it to the desired mass and tidal/half mass radius scaling velocities accordingly
            self.initialize_king_model(n_particles, M_cluster, W0, r_tidal, r_half)
        if unbound_particles:
            self.unbound.particles.add_particles(unbound_particles)
        if field_code_mode == 'center_of_mass':
            self.field_code_mode = 'center_of_mass'
            self.center_of_mass=center_of_mass(self.bound.particles)
        else:
            self.field_code_mode = 'direct'
            if field_code_mode!='direct':
                print('ERROR: you must choose either direct or center_of_mass for field_code_mode. Defaulting to direct.')
            
            self.field_code=field_code
            self.field_code_number_of_workers=field_code_number_of_workers
            # initialize the code for get_gravity_at_point and get_potential_at_point
            self.gravity_from_cluster = bridge.CalculateFieldForCodes(
                self.new_code_to_calculate_gravity,               
                input_codes=[self.bound],                       
                )
        
        # evolve to 0 Myr so we have dt_soft set
        self.bound.evolve_model(0 | units.Myr)
        # initialize stellar evolution
        self.stellar_evolution=None
        if stellar_evolution:
            self.stellar_evolution = stellar_evolution()
            self.stellar_evolution.particles.add_particles(self.bound.particles)
            self.channel_from_stellar_evolution = self.stellar_evolution.particles.new_channel_to(self.bound.particles, attributes=['mass', 'radius'])

    def new_code_to_calculate_gravity(self): 
            result = self.field_code(self.converter, number_of_workers=self.field_code_number_of_workers, mode='cpu')  # this can be GPU based at some point
            return result
    # initialize the king model
    def initialize_king_model(self, n_particles, M_cluster, W0, r_tidal=None | units.pc, r_half=None | units.pc):
        # we either fix the number of stars, or the total mass (down to stochastic fluctuations)
        m_stars = new_masses(stellar_mass=M_cluster,number_of_stars=n_particles, upper_mass_limit=100.0 | units.MSun,lower_mass_limit=0.1 | units.MSun)
        cluster = new_physical_king_model(W0, masses=m_stars, tidal_radius=r_tidal, half_mass_radius=r_half)
        self.bound.particles.add_particles(cluster)

    # get the gravity at a point
    def get_gravity_at_point(self,radius,x,y,z):
        if self.field_code_mode == 'center_of_mass':
            ax,ay,az=self.center_of_mass.get_gravity_at_point(self.bound.particles.LagrangianRadii(mf=[0.5])[0][0].as_vector_with_length(len(x)),x,y,z) # here we should set radius automatically to the half mass radius (if it was plummer) - maybe diff for king?
        elif self.field_code_mode == 'direct':
            ax,ay,az=self.gravity_from_cluster.get_gravity_at_point(radius,x,y,z)
        return ax,ay,az
    
    # get the potential at a point
    def get_potential_at_point(self,radius,x,y,z):
        if self.field_code_mode == 'center_of_mass':
            phi=self.center_of_mass.get_potential_at_point(radius,x,y,z)
        elif self.field_code_mode == 'direct':
            phi = self.gravity_from_cluster.get_potential_at_point(radius,x,y,z)
        return phi
    
    # evolve the bound particles
    def evolve_model(self,tend):
        if self.stellar_evolution:
            adjusted_dt = False
            minimum_value_of_dt_soft = self.bound.parameters.dt_soft
            dt = 2.*self.stellar_evolution.particles.time_step.min()
            while dt<(tend-self.bound.model_time):
                dt_nbody = self.converter.to_nbody(dt).number
                n_min_time_step = math.ceil(math.log(dt_nbody, 0.5))
                dt = self.converter.to_si(0.5**n_min_time_step | nbody_system.time)

                print('evolving to', (self.bound.model_time+dt).in_(units.Myr))
                self.stellar_evolution.evolve_model(self.bound.model_time+dt/2)
                self.channel_from_stellar_evolution.copy()

                # may need to adjust dt_soft in petar to capture this!
                if dt < self.bound.parameters.dt_soft:
                    adjusted_dt = True
                    self.bound.parameters.dt_soft=dt
                    minimum_value_of_dt_soft = min(dt, minimum_value_of_dt_soft)
                    print('adjusting dt_soft', dt.in_(units.Myr))
                    print('actual value', self.bound.parameters.dt_soft.in_(units.Myr))
                self.bound.evolve_model(self.bound.model_time+dt)
                if adjusted_dt:
                    self.bound.parameters.dt_soft=0 | units.Myr
                    self.bound.evolve_model(self.bound.model_time)
                    print('reset dt_soft to', self.bound.parameters.dt_soft.in_(units.Myr))
                    adjusted_dt=False

                self.stellar_evolution.evolve_model(self.bound.model_time)
                self.channel_from_stellar_evolution.copy()
                print(self.bound.model_time.in_(units.Myr))
                dt = 2.*self.stellar_evolution.particles.time_step.min()
            print('exited while loop')
            self.bound.parameters.dt_soft=minimum_value_of_dt_soft
            print(self.bound.parameters.dt_soft.in_(units.Myr))
            self.stellar_evolution.evolve_model(self.bound.model_time+(tend-self.bound.model_time)/2)
            self.channel_from_stellar_evolution.copy()
            print('about to evolve petar')
            self.bound.evolve_model(tend)
            print('evolved petar')
            self.stellar_evolution.evolve_model(tend)
            self.channel_from_stellar_evolution.copy()
            self.bound.parameters.dt_soft=0 | units.Myr
            print('at end of evolve_model')
        else:
            self.bound.evolve_model(tend)

    def transfer_unbound_particles(self):
        bound = self.bound.particles.bound_subset(unit_converter=self.converter,tidal_radius=self.bound.particles.LagrangianRadii(mf=[0.95])[0][0], strict=True)
        new_unbound = self.bound.particles.difference(bound).copy()
        self.unbound.particles.add_particles(new_unbound)
        self.bound.particles.remove_particles(new_unbound)
        if self.stellar_evolution:
            # redefine channel just in case?
            new_unbound=self.stellar_evolution.particles.difference(bound).copy()
            self.unbound.stellar_evolution.particles.add_particles(new_unbound) # here we want to add the equivalent SE particles (with age and other properties!), not the dynamical ones
            self.unbound.channel_from_stellar_evolution = self.stellar_evolution.particles.new_channel_to(self.unbound.particles, attributes=['mass', 'radius'])
            self.stellar_evolution.particles.remove_particles(new_unbound) # will this remove the correct particles?
            
    
    @property
    def all_particles(self):
        return ParticlesSuperset([self.bound.particles, self.unbound.particles])
    
    # has to only return cluster stars so these are kicked by bridge. Add unbound stars seperately 
    @property
    def particles(self):
        return self.bound.particles
    

# a class to evolve the unbound star particles - allows us to place them in bridge seperately
class drifter(object):
    """
    unbound_stars=unbound_stars(initialization_params)
    derived system, represents unbound star particles
    provides: particles, evolve_model
    """
    def __init__(self, particles=Particles(), initial_time=zero, stellar_evolution=None):
        # initialize unbound particles here
        self.particles = particles
        self.model_time = initial_time
        self.stellar_evolution=None
        if stellar_evolution:
            self.stellar_evolution = stellar_evolution()
            if len(self.particles) > 0:
                self.stellar_evolution.particles.add_particles(self.particles)
                self.channel_from_stellar_evolution = self.stellar_evolution.particles.new_channel_to(self.particles, attributes=['mass', 'radius'])
        
    def evolve_model(self, tend):
        # evolve the unbound particles here
        if len(self.particles) > 0:
            delta_t=tend-self.model_time
            if self.stellar_evolution:
                # since nothing else knows about the drifters we can just evolve a full step (as long as we don't have SN kicks)
                while self.model_time < tend:
                    dt = min(2.*self.stellar_evolution.particles.time_step.min(), tend-self.model_time)
                    self.stellar_evolution.evolve_model(self.model_time+dt/2)
                    self.channel_from_stellar_evolution.copy()
                    self.particles.position += self.particles.velocity * dt
                    self.stellar_evolution.evolve_model(self.model_time+dt)
                    self.channel_from_stellar_evolution.copy()
                    self.model_time +=dt
            else:
                self.particles.position += self.particles.velocity * delta_t
                self.model_time = tend
        