import numpy as np
from scipy.integrate import quad
import math
from amuse.units import constants, units
from amuse.support.literature import LiteratureReferencesMixIn
from amuse.support.exceptions import AmuseWarning

####################################################################
# Class for computing dynamical friction for a given (spherical) static potential
####################################################################
# To Do:
# - non spherical potentials...
class dynamical_friction():
    """
    Application of the Chandrasekhar Dynamical friction formula following Petts J. A., Read J. I., Gualandris A., 2016, MNRAS,463,858.
    
    :argument density_model: a class with methods mass_density(r), log_log_slope(r) and circular_velocity(r) for the background
    :argument code: the gravity (or drift) code containing the satellite
    :argument r_half: the half mass radius of the satellite - method!
    """
    def __init__(self, density_model, particles, half_mass_radius, G=constants.G):
        self.density_model = density_model
        self.particles=particles # we need to be able to access latest version of the particles... ideally without storing twice
        self.G = G
        self.half_mass_radius = half_mass_radius

    def set_rv_mass(self):
        """update the satellite properties in the dynamical friction model
        """
        self.x, self.y, self.z = self.particles.center_of_mass()
        self.velocity= self.particles.center_of_mass_velocity()
        self.r = np.sqrt(self.x ** 2.0 + self.y ** 2.0 + self.z ** 2.0)
        self.mass = self.particles.mass.sum()

    def get_gravity_at_point(self,eps,x,y,z):
        accel_dynamical = self.dynamical_friction()
        ax = accel_dynamical[0]
        ay = accel_dynamical[1]
        az = accel_dynamical[2]
        return ax,ay,az
    
    def dynamical_friction(self): 
        self.set_rv_mass()
        gamma = self.density_model.log_log_slope(self.r)
        Lambda = self.r.value_in(units.pc)*min(1.,1./gamma)/max(self.half_mass_radius().value_in(units.pc),(self.G*self.mass/self.velocity.length()**2).value_in(units.pc)) 
        coulomb_log = np.log(1 + Lambda**2)
        sigma = self.sigmar()
        return -2*np.pi*coulomb_log*self.G**2 *self.mass*self.density_model.mass_density(self.r)*self.velocity/self.velocity.length()**3 *self.thermal_integral(self.velocity.length()/(2**.5*sigma))

    # solve the spherical jeans equation for velocity dispersion
    def sigmar(self):
        r_kpc = self.r.value_in(units.kpc)
        return ((quad(lambda x: self.density_model.mass_density(x | units.kpc).value_in(units.MSun/units.kpc**3) *
                               (self.density_model.circular_velocity(x | units.kpc)**2).value_in(units.kpc**2/units.s**2)/x,
                                 r_kpc, np.inf,)[0]/ self.density_model.mass_density(self.r).value_in(units.MSun/units.kpc**3)) | units.kpc**2/units.s**2).sqrt()

    def thermal_integral(self, x):
        return math.erf(x) - 2*x/np.pi**.5 * np.exp(-x**2)
    