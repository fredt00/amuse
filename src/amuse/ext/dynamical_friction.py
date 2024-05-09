import numpy as np
from scipy.integrate import quad
import math
from amuse.units import constants, units
from amuse.support.literature import LiteratureReferencesMixIn
from amuse.support.exceptions import AmuseWarning

####################################################################
# Class for computing dynamical friction for a given (spherical) static potential
####################################################################

class dynamical_friction():
    """
    Application of the Chandrasekhar Dynamical friction formula following Petts J. A., Read J. I., Gualandris A., 2016, MNRAS,463,858.
    
    :argument density_model: a class with methods mass_density(r), log_log_slope(r) and circular_velocity(r) for the background
    :argument code: the gravity (or drift) code containing the satellite
    :argument r_half: the half mass radius of the satellite to be used in the case of a point mass
    """
    def __init__(self, density_model, code, G=constants.G, r_half=4.35 | units.pc):
        self.density_model = density_model
        self.code=code # we need to be able to access latest version of the particles... ideally without storing twice
        self.G = G
        self.r_half = r_half

    def set_rv_mass(self):
        """update the satellite properties in the dynamical friction model
        """
        self.x, self.y, self.z = self.code.particles.center_of_mass()
        self.velocity= self.code.particles.center_of_mass_velocity()
        self.r = np.sqrt(self.x ** 2.0 + self.y ** 2.0 + self.z ** 2.0)

        # find the current half mass radius - slow!
        if len(self.code.particles)!=1:
            self.r_half = self.code.particles.LagrangianRadii(mf=[0.5])[0][0]
        self.mass = self.code.particles.mass.sum()

    def get_gravity_at_point(self,eps,x,y,z):
        accel_dynamical = self.dynamical_friction()
        ax = accel_dynamical[0]
        ay = accel_dynamical[1]
        az = accel_dynamical[2]
        return ax,ay,az
    
    def dynamical_friction(self): 
        self.set_rv_mass()
        gamma = self.density_model.log_log_slope(self.r)
        Lambda = self.r.value_in(units.pc)*min(1.,1./gamma)/max(self.r_half.value_in(units.pc),(self.G*self.mass/self.velocity.length()**2).value_in(units.pc)) 
        coulomb_log = np.log(1 + Lambda**2)
        sigma = self.sigmar(self.r)
        return -2*np.pi*coulomb_log*self.G**2 *self.mass*self.density_model.mass_density(self.r)*self.velocity/self.velocity.length()**3 *self.thermal_integral(self.velocity.length()/(2**.5*sigma))

    # solve the spherical jeans equation for velocity dispersion
    def sigmar(self, r):
        r_kpc = r.value_in(units.kpc)
        return ((quad(lambda x: self.density_model.mass_density(x | units.kpc).value_in(units.MSun/units.kpc**3) *
                               (self.density_model.circular_velocity(x | units.kpc)**2).value_in(units.kpc**2/units.s**2)/x,
                                 r_kpc, np.inf,)[0]/ self.density_model.mass_density(r).value_in(units.MSun/units.kpc**3)) | units.kpc**2/units.s**2).sqrt()

    def thermal_integral(self, x):
        return math.erf(x) - 2*x/np.pi**.5 * np.exp(-x**2)
    
    @property
    def particles(self):
        return self.code.particles
    

class NFW_profile(LiteratureReferencesMixIn):
    """
    Our own NFW profile that has additional functions for use in dynamical friction: log_log_slope
    * density(r) = rho0 / [r/rs * (1+r/rs)^2], where is the spherical radius
    * potential(r) = -4*pi*G*rho0*rs^2 * ln(1+r/rs)/(r/rs)
    
    .. [#] Navarro, Julio F.; Frenk, Carlos S.; White, Simon D. M., The Astrophysical Journal, Volume 490, Issue 2, pp. 493-508 (1996)
    
    :argument rho0: density parameter
    :argument rs: scale radius
    """
    def __init__(self,rho0,rs,G=constants.G):
        LiteratureReferencesMixIn.__init__(self)
        self.rho0 = rho0
        self.rs = rs
        self.G = G
        self.four_pi_rho0 = 4.*np.pi*self.rho0
        self.four_pi_rho0_G = self.four_pi_rho0*self.G
    
    def radial_force(self,r):
        r_rs = r/self.rs
        ar = self.four_pi_rho0_G*self.rs**3*(1./(r*self.rs+r**2)-(1./r**2)*np.log(1.+r_rs))
        return ar
    def log_log_slope(self, r):
        return (1 + 3*r/self.rs)/(1+r/self.rs)
    
    def get_potential_at_point(self,eps,x,y,z):
        r = (x**2+y**2+z**2).sqrt()
        r_rs = r/self.rs
        return -1.*self.four_pi_rho0_G*self.rs**2*np.log(1.+r_rs)/r_rs
    
    def get_gravity_at_point(self,eps,x,y,z):
        r = (x**2+y**2+z**2).sqrt()
        fr = self.radial_force(r)
        ax = fr*x/r
        ay = fr*y/r
        az = fr*z/r
        return ax,ay,az
    
    def enclosed_mass(self,r):
        fr = self.radial_force(r)
        return -r**2/self.G*fr
    
    def circular_velocity(self,r):
        fr = self.radial_force(r)
        return (-r*fr).sqrt()
    
    def mass_density(self,r):
        r_rs = r/self.rs
        return self.rho0 / (r_rs*(1.+r_rs)**2)