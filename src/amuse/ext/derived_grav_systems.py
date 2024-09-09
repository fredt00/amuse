from amuse.units import constants
from amuse.datamodel import Particles, ParticlesSuperset
from amuse.ic.kingmodel import new_physical_king_model
from amuse.ic.brokenimf import new_masses
from amuse.couple import bridge
from amuse.units.quantities import zero
from amuse.units import units
import math
from amuse.units import nbody_system
import numpy as np
from scipy.integrate import simpson as simp
from amuse.ic.brokenimf import MultiplePartIMF
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

# class for computing tidal fields
class tidal_field(object):
    """
    tidal_field=tidal_field(grav_instance)
    derived system, returns tidal field system with get_tidalfield_at_point
    get_tidalfield_at_point_per_gyr_sq, and tidal_radius methods
    """
    def __init__(self,grav_instance):
        self.grav_instance=grav_instance

    def get_tidalfield_at_point(self,scale,x,y,z):
        # perhaps this could vary, = self.rhalf
        h = scale
        ax0,ay0,az0 = self.grav_instance.get_gravity_at_point(0 | units.pc, x, y, z)
        axx,ayx,azx = self.grav_instance.get_gravity_at_point(0 | units.pc, x+h, y, z)
        axy,ayy,azy = self.grav_instance.get_gravity_at_point(0 | units.pc, x, y+h, z)
        axz,ayz,azz = self.grav_instance.get_gravity_at_point(0 | units.pc, x, y, z+h)
        Txx = ((axx-ax0)/h)
        Tyy = ((ayy-ay0)/h)
        Tzz = ((azz-az0)/h)

        Txy = ((ayx-ay0)/h)
        Txz = ((azx-az0)/h)
        Tyz = ((azy-az0)/h)
        return Txx,Tyy,Tzz,Txy,Txz,Tyz
    
    def get_tidalfield_at_point_per_gyr_sq(self,scale,x,y,z):
        h = scale
        Txx,Tyy,Tzz,Txy,Txz,Tyz=self.get_tidalfield_at_point(h,x,y,z)
        Txx=Txx.value_in(units.gyr**-2)
        Tyy=Tyy.value_in(units.gyr**-2)
        Tzz=Tzz.value_in(units.gyr**-2)
        Txy=Txy.value_in(units.gyr**-2)
        Txz=Txz.value_in(units.gyr**-2)
        Tyz=Tyz.value_in(units.gyr**-2)
        return Txx,Tyy,Tzz,Txy,Txz,Tyz
    
    def tidal_radius(self, scale, x, y, z, satellite_mass):
        eigenvalues = self.tidal_tensor_eigenvalues(scale, x, y, z)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        omegasq = np.abs(eigenvalues.sum())/3
        T = max_eigenvalue + omegasq
        return (constants.G * satellite_mass/T)**(1/3)
    
    def tidal_tensor_eigenvalues(self, scale, x, y, z):
        Txx, Tyy, Tzz, Txy, Txz, Tyz = self.get_tidalfield_at_point_per_gyr_sq(scale, x, y, z)
        tidal_tensor = np.array([[Txx, Txy, Txz],
                                [Txy, Tyy, Tyz],
                                [Txz, Tyz, Tzz]])
        eigenvalues, _ = np.linalg.eig(tidal_tensor)
        eigenvalues = eigenvalues | units.gyr**-2
        return eigenvalues

# create a wrapper class for a gravity code to describe a star cluster including bound and unbound particles and stellar evolution
class star_cluster(tidal_field):
    """
    star_cluster=star_cluster(grav_instance,converter)
    derived system, returns star cluster system with
    get_gravity_at_point, get_potential_at_point reimplemented in 
    base_class
    """
    def __init__(self,code,code_converter, particles=None, W0=5, r_tidal=None | units.pc,r_half=None | units.pc, n_particles=None,
                  M_cluster=False, code_number_of_workers=1, stellar_evolution = None, field_code = None, time= 0 | units.Myr):
        # inherit the tidal field stuff
        super().__init__(field_code)
        
        # initialize converter from SI to Nbody units
        self.converter=code_converter
        # initialize the code for handling bound cluster particles (collisional)
        self.bound=code(self.converter, mode='openmp',number_of_workers=code_number_of_workers)
        # initialize the code for handling unbound particles (collisionless)
        self.unbound = drifter()

        # unsure if we can set bound.model_time, etc so we will have a self.model_time that we know we can control
        self.model_time = time
        self.bound.parameters.begin_time = time 
        self.unbound.model_time = time

        # if restarting, add the particles to respective codes
        if particles:
            self.particles = particles
        else:
        # create a scale free king model,then scale it to the desired mass and tidal/half mass radius scaling velocities accordingly
            cluster = self.initialize_king_model(n_particles, M_cluster, W0, r_tidal, r_half)
            self.particles = Particles()
            self.particles.add_particles(cluster)
        if time==0 | units.Myr:
            # define a particle attribute to keeping track of escaping stars. if this is true in prev timestep, we remove
            # the particle if it is still unbound in the one being considered - hopefully remove some shot noise in removal
            self.particles.escape_flag = False
            # this keeps track of what particles are in the unbound code and when they were added
            self.particles.unbound_time = -1 | units.Myr
        

        self.bound.particles.add_particles(self.particles[self.particles.unbound_time<0 | units.Myr])
        self.unbound.particles.add_particles(self.particles.difference(self.bound.particles))
        self.center_of_mass=center_of_mass(self.bound.particles)
  
        # evolve to 0 Myr so we have dt_soft set
        self.bound.evolve_model(0 | units.Myr)

        # initialize channels for copying data to the framework
        self.b2f = self.bound.particles.new_channel_to(self.particles, attributes=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.u2f = self.unbound.particles.new_channel_to(self.particles, attributes=['x', 'y', 'z', 'vx', 'vy', 'vz'])

        # channels for copying dynamical quantities altered by SE from framework to dynamics codes
        # also must copy pos vel in case of bridge kicking
        self.f2b = self.particles.new_channel_to(self.bound.particles, attributes=['mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.f2u = self.particles.new_channel_to(self.unbound.particles, attributes=['mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

         # initialize stellar evolution
        self.stellar_evolution=None
        if stellar_evolution:
            self.stellar_evolution = stellar_evolution()
            self.stellar_evolution.model_time = time
            self.stellar_evolution.particles.add_particles(self.particles)
            # note - it is important that all required restart attributes are copied to the framework particles from SE
            self.s2f = self.stellar_evolution.particles.new_channel_to(self.particles)#, attributes=['mass', 'radius'])
            self.f2s = self.particles.new_channel_to(self.stellar_evolution.particles, attributes=["mass", "radius", "x", "y", "z", "vx", "vy", "vz"])
            self.s2f.copy()

    def new_code_to_calculate_gravity(self): 
        result = self.field_code(self.converter, number_of_workers=self.field_code_number_of_workers, mode='cpu')  # this can be GPU based at some point
        return result
    
    # initialize the king model
    def initialize_king_model(self, n_particles, M_cluster, W0, r_tidal=None | units.pc, r_half=None | units.pc):
        # we either fix the number of stars, or the total mass (down to stochastic fluctuations)
        m_stars = new_masses(stellar_mass=M_cluster,number_of_stars=n_particles, upper_mass_limit=100.0 | units.MSun,lower_mass_limit=0.1 | units.MSun)
        cluster = new_physical_king_model(W0, masses=m_stars, tidal_radius=r_tidal, half_mass_radius=r_half)
        return cluster

    def half_mass_radius(self):
        return self.bound.particles.LagrangianRadii(mf=[0.5])[0][0]
    
    # get the gravity at a point
    def get_gravity_at_point(self,radius,x,y,z):
        ax,ay,az=self.center_of_mass.get_gravity_at_point(self.half_mass_radius().as_vector_with_length(len(x)),x,y,z) 
        return ax,ay,az
    
    # get the potential at a point
    def get_potential_at_point(self,radius,x,y,z):
        return self.center_of_mass.get_potential_at_point(radius,x,y,z)
    
    # evolve the bound particles
    def evolve_model(self,tend):
        if self.stellar_evolution:
            dt = tend - self.bound.model_time

            self.f2s.copy()
            self.stellar_evolution.evolve_model(self.bound.model_time+dt/2)
            self.s2f.copy()

            self.f2b.copy()
            self.bound.evolve_model(tend)
            self.b2f.copy()

            self.f2s.copy()
            self.stellar_evolution.evolve_model(self.bound.model_time)
            self.s2f.copy()

            # Below is for a variable timestep - does not sees necessaty

            # here we need to stay in int (the exponent of 0.5) until calls to dynamics and SE to avoid floating point errors
            # also it seems petar can't handle dt_soft below a certain value for a given system - perhaps when we approach similar timesteps
            # to the hermite scheme or binary periods? or it could just be rounding errors from all the conversions going on
            # either way we will set the minimum timestep to 0.5**15 for now (this does depend on the converter used though)
            # maximum_n_allowed = 17
            # initial_n_for_dt_soft = math.ceil(math.log(self.converter.to_nbody(self.bound.parameters.dt_soft).number, 0.5))
            # maximum_n_for_dt_soft = initial_n_for_dt_soft
            # dt = self.stellar_evolution.particles.time_step.min()
            # # change this to evolve stellar evolution until mass has changed by 1-10% or we reach the bridge time step
            # while dt<(tend-self.bound.model_time):
            #     dt_se_nbody = self.converter.to_nbody(dt).number
            #     n_min_se_time_step = math.ceil(math.log(dt_se_nbody, 0.5))
            #     if n_min_se_time_step > maximum_n_allowed:
            #         n_min_se_time_step = maximum_n_allowed
            #     dt = self.converter.to_si(0.5**n_min_se_time_step | nbody_system.time)
            #     self.stellar_evolution.evolve_model(self.bound.model_time+dt/2)
            #     self.s2f.copy()
                

            #     # may need to adjust dt_soft in petar to capture this timescale - stay in integer exponent!
            #     if n_min_se_time_step > initial_n_for_dt_soft:
            #         self.bound.parameters.dt_soft=dt # petar can take nbody or SI units
            #         maximum_n_for_dt_soft = max(n_min_se_time_step, maximum_n_for_dt_soft)
            #         current_n_for_dt_soft = n_min_se_time_step
            #     else:
            #         self.bound.parameters.dt_soft=0.5**initial_n_for_dt_soft | nbody_system.time

            #     self.f2b.copy()
            #     self.bound.evolve_model(self.bound.model_time+dt)
            #     self.b2f.copy()

            #     self.stellar_evolution.evolve_model(self.bound.model_time)
            #     self.s2f.copy()

            #     dt = self.stellar_evolution.particles.time_step.min()

            # remaining_time = tend-self.bound.model_time
            # self.stellar_evolution.evolve_model(self.bound.model_time+remaining_time/2)
            # self.s2f.copy()
   
            # self.bound.parameters.dt_soft = self.converter.to_si(0.5**maximum_n_for_dt_soft | nbody_system.time)
            # self.f2b.copy()
            # self.bound.evolve_model(self.bound.model_time+remaining_time)
            # self.b2f.copy()

            # self.bound.parameters.dt_soft=0 | units.Myr
            # self.bound.evolve_model(self.bound.model_time)
            # self.stellar_evolution.evolve_model(tend)
            # self.s2f.copy()
        else:

            self.f2b.copy()
            self.bound.evolve_model(tend)
            self.b2f.copy()

        self.f2u.copy()
        self.unbound._evolve_model(tend) # update the unbound particles - this should work ok in bridge because evolves happen after kicking
        self.u2f.copy()
        self.model_time = tend

    def transfer_unbound_particles(self):
        # transfer unbound particles to the unbound code
        current_framework_bound = self.particles[self.particles.unbound_time<0 | units.Myr]
        CoM = current_framework_bound.center_of_mass()
        bound_subset = current_framework_bound.bound_subset(unit_converter=self.converter,tidal_radius=self.tidal_radius(4|units.pc, CoM.x, CoM.y, CoM.z, current_framework_bound.total_mass()), strict=True)
        new_unbound = current_framework_bound.difference(bound_subset)
        remove=new_unbound#[new_unbound.escape_flag]# remove only particles not already removed
        # update escape flag for particles that were not unbound last tstep but are now
        # new_unbound.escape_flag = True
        remove.unbound_time = self.model_time
        self.unbound.particles.add_particles(remove)
        self.bound.particles.remove_particles(remove)
        # redeifine channel just in case?
        self.u2f = self.unbound.particles.new_channel_to(self.particles, attributes=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.b2f = self.bound.particles.new_channel_to(self.particles, attributes=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.f2b = self.particles.new_channel_to(self.bound.particles, attributes=['mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.f2u = self.particles.new_channel_to(self.unbound.particles, attributes=['mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    
    def stop(self):
        self.bound.stop()
        self.unbound.stop()
        if self.stellar_evolution:
            self.stellar_evolution.stop()

# a class to evolve the unbound star particles - allows us to place them in bridge seperately
class drifter(object):
    """
    unbound_stars=unbound_stars(initialization_params)
    derived system, represents unbound star particles
    provides: particles, evolve_model
    """
    def __init__(self, particles=Particles(), initial_time=zero):
        # initialize unbound particles here
        self.particles = particles
        self.model_time = initial_time
    
    def evolve_model(self, tend):
        # dummy evolve model so we can add this to bridge - really evolve is carried out in star_cluster
        # this is needed so that stream and cluster can kick each other - but we need them to evolve in sequence still
        pass

    def _evolve_model(self, tend):
        # evolve the unbound particles here
        if len(self.particles) > 0:
            dt = tend - self.model_time
            self.particles.position += self.particles.velocity * dt
            self.model_time = tend

    def stop(self):
        pass
