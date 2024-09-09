#####################################
# Cluster model particle for use in AMUSE
# currently the EMACSS model is implemented - works well for tidally filling GCs...
# if SE is off and tidally underfilling GCs, the model will not work well
#####################################
from amuse.units import constants
from amuse.datamodel import Particles
from amuse.units import units
import numpy as np
from scipy.integrate import simpson
from amuse.ext.derived_grav_systems import tidal_field

global xi0
xi0 = 0.0075 #0.0142 for equal mass, 0.0075 for the paper
global zeta
zeta = 0.1 # reducing this helps match the mass loss profile, but then the half mass radius does not increase enough
# as mu ~epsilon = zeta after nc
global N1
N1 = 1000 # 15000 and for equal mass clusters
global R1
R1 =0.22
global gamma_c
gamma_c = 0.02 # 0.11 for equal mass clusters
global f
f=0.3
global z
z=2 #1.61 for equal mass clusters
global kapppa
kappa_1=0.24 # ours seems to go to 0.9?
# chi between 0 and 1, depends on mass of escaping stars either mlow or mbar
global Chi
Chi= 0.55#0.35+(1-0.35)*(self.half_mass_radius/rtidal)**1.1#0.3#0.55 in paper, we find for circular orbit we have to set 0.45
global q
q=2 # doesn't seem very sensitive to this
global M1
M1=4
global nc
nc =12.5 # 12.5 in paper ... maybe lowerfor us? 5?


global tidal_shock_energy_fraction
tidal_shock_energy_fraction =0.25 # - could be more physically motivated? written in terms of other parameters?

# stellar evolution parameters
global main_sequence_lifetime_m_up
main_sequence_lifetime_m_up = 3.3 | units.Myr
global m_up_inf
m_up_inf = 1.2 | units.MSun
global a
a=-2.7

global psi1
psi1=8.0 # this is different because mup is 100 for them!
global psi0
psi0 = 1.6

global nu 
nu = 0.07
global Y
Y = 90
global b
b=1.35
global y 
y = -0.3

class internal_dynamics(tidal_field):
    def __init__(self, N, mbar, mbar_se, half_mass_radius, kappa, M_seg, n_trhp, particles, grav_instance, stellar_evolution, VG, time):
        
        super().__init__(grav_instance)
        
        self.VG = VG

        self.particles = particles

        if not half_mass_radius:
            # set to RV filling - depends on cluster W0
            half_mass_radius= self.rtidal()* 0.19

        self.N=N
        self.mbar=mbar
        self.mbar_se = mbar
        self.mbar_init = mbar
        self.half_mass_radius=half_mass_radius
        self.kappa=kappa
        self.M_seg=M_seg
        self.m_low = 0.1 | units.MSun
        # self.m_up = 100 | units.MSun
        self.m_max = 100 | units.MSun
        # self.psi=14#8.0#13.5364073081 # this should depend on mass spectrum WITHIN half_mass_radius   - !!!NOTE possibly this should be the case for many parameters including shape parameter and mean mass etc...
        # ok so i think we want mean mass to be for the whole cluster, buy psi should depend only within half_mass_radius... makes it harder to relate to other variables like mbar...
        # perhaps M_seg and or kappa could relate mbar_h to mbar?
        self.n_trhp = 0
        
        self.stellar_evolution = stellar_evolution

        self.model_time = time
        
        # compute the initial relaxation time
        self.trhp = self.relaxation_time_prime()

    # singular isothermal sphere - used if VG is specified. for comparisson with EMACSS paper
    def emacss_isothermal_rj(self):
        RG= self.particles.position.lengths()[0]
        MG = RG*self.VG**2/constants.G#9.53e+10 | units.MSun
        return pow((self.N*self.mbar)/(2.0*MG),(1.0/3.0))*RG

    def rtidal(self):
        if self.grav_instance:
            return self.tidal_radius(40 | units.pc, self.particles.position[0].x, self.particles.position[0].y,
                                        self.particles.position[0].z, self.particles.mass[0])
        elif self.VG:
            return self.emacss_isothermal_rj()
        else: 
            return np.inf | units.pc
            
    # stellar evolution quantities
    def main_sequence_lifetime(self,mass):
        return main_sequence_lifetime_m_up * (1 + np.log(mass/self.m_max)/np.log(self.m_max/m_up_inf))**a
    
    def f_ind(self):
        if self.RhJ()>R1:
            return Y * (self.RhJ() - R1)**b
        else:
            return 0
    
    def m_up(self):
        m_up = self.m_max
        if self.model_time > main_sequence_lifetime_m_up:
           m_up = self.m_max*pow(self.m_max/m_up_inf,pow(main_sequence_lifetime_m_up/self.model_time,0.37) - 1.0); 
        m_up = np.sqrt(pow(m_up,2)+pow(m_up_inf,2))
        return m_up
        
    def psi(self):
        if self.model_time < main_sequence_lifetime_m_up:
            return psi1
        else:
            return (psi1-psi0)*(self.model_time/self.main_sequence_lifetime(self.m_max))**y + psi0 # possibly needs to be t_se(t)
        
    
    # derived quantities
    def relaxation_time(self):
        return 0.138 * np.sqrt(self.N * self.half_mass_radius**3/(constants.G*self.mbar)) /np.log(gamma_c * self.N)
    
    def relaxation_time_prime(self):
        return self.relaxation_time()/self.psi()
    
    def total_energy(self):
        return -self.kappa * constants.G*(self.N*self.mbar)**2./self.half_mass_radius

    def RhJ(self):
        return self.half_mass_radius/self.rtidal()

    # note! here they use rv/rj not rh/rj!! and rv=rh/4kappa
    def P(self):
        return (self.RhJ()/(R1*4*self.kappa))**z * ((self.N*np.log(gamma_c*N1))/(N1*np.log(gamma_c*self.N)))**(1-0.75)
    
    def F(self):
        F=0.
        if nc/2<self.n_trhp <= nc:
            F = 2.*self.n_trhp/nc - 1.
        if self.n_trhp > nc:
            F=1.
        return F
    
    # dimensionless rates
    def xi(self):
        return self.xi_i() + self.xi_e()
    def xi_i(self):
        if self.stellar_evolution:
            return self.f_ind()*self.gamma_se()  # here is an inconsistency in the paper1
        else:
            return 0
    def xi_e(self):
        return self.F() * xi0 * (1 - self.P()) + (f + (1 - f) * self.F()) * 3/5 * zeta * self.P()
    
    def gamma(self):
        return self.gamma_dyn() - self.gamma_se()
    
    def gamma_dyn(self):
        return (1 - self.mesc()/self.mbar) * self.S() * self.U() * self.xi()
    
    # note sign change compared to paper
    def gamma_se(self):
        if self.stellar_evolution and self.model_time>main_sequence_lifetime_m_up:
            return nu * self.trhp/self.model_time * self.mbar_se/self.mbar
        else:
            return 0.

    def lambd(self):
        lambd=0.
        if self.n_trhp > 0.5*nc:
            lambd += (kappa_1 - self.kappa) * (2*self.n_trhp/nc - 1)
        return lambd
    
    def epsilon(self):
        epsilon=0.
        if self.n_trhp > nc:
            epsilon=zeta
        elif self.model_time > main_sequence_lifetime_m_up:
            epsilon = self.M_seg*self.gamma_se() + self.tidal_escape()
        else:
            epsilon = self.tidal_escape()
        return epsilon
    
    def tidal_escape(self):
        return self.RhJ()/self.kappa * self.xi() * self.mesc()/self.mbar

    def mu(self):
        return self.epsilon() - 2 * self.xi() + 2 * self.gamma() + self.lambd()
    
    # other derived quantities
    def mesc(self):
        return Chi * (self.mbar - self.m_low) + self.m_low
    
    def S(self):
        return ((self.M_seg-3.)/(M1-3.))**q
    
    def U(self):
        return np.abs(self.m_up() - self.mbar)/self.m_up() # should this be m_up(t) or m_max which is fixed???
    # fundamentally we should not be counting stars that have evolved to lose mass?

    # derived rates divided by variable (so log rate)
    def dNdt(self):
        return -self.xi() * self.N/self.trhp
    def dmbardt(self):
        return self.gamma() * self.mbar/ self.trhp
    def dMsegdt(self):
        return (M1-self.M_seg) * self.M_seg/self.trhp
    def dkdt(self):
        return self.lambd() * self.kappa/ self.trhp
    def drdt(self):
        return self.mu() * self.half_mass_radius/self.trhp
    def dmbar_se_dt(self):
        return -self.gamma_se() * self.mbar/ self.trhp

    def dtrhpdt(self):
        # for counting
        return 1./self.trhp
    
    # def dpsidt(self):
        #return -5/2*(1-self.F())*(self.dmbardt()/self.M_seg-self.dMsegdt()/self.M_seg**2)
        # return -5/2*(self.dmbardt() * (self.psi-7.8)-(1-self.F())*self.dMsegdt()/self.M_seg**2)
    
    def min_step(self):
        return 1.0/(1e6/self.trhp+1e6/self.model_time)

    # an array containing all the evolved parameters to let us update them simultaneously 
    def get_nbody(self):
        return np.array([self.model_time, self.N, self.mbar, self.mbar_se, self.half_mass_radius, self.n_trhp, self.kappa, self.M_seg])
    
    def set_nbody(self, nbody):
        self.model_time = nbody[0]
        self.N = nbody[1]
        self.mbar = nbody[2]
        self.mbar_se = nbody[3]
        self.half_mass_radius = nbody[4]
        self.n_trhp = nbody[5]
        self.kappa = nbody[6]
        self.M_seg = nbody[7]
    
    def rate_array(self):
        return np.array([1.0, self.dNdt(), self.dmbardt(), self.dmbar_se_dt(), self.drdt(), self.dtrhpdt(), self.dkdt(), self.dMsegdt()])


## TO DO
# - add a unit converter to the class
class star_cluster_particle(internal_dynamics):
    def __init__(self, N=None, mass=None, half_mass_radius=4.35 | units.pc, kappa=0.2, M_seg=3, mbar=None, mbar_se=None, n_trhp=0,
                 position=None, velocity=None, grav_instance=None, stellar_evolution=True, VG=None, time = 0 | units.Myr):
        
        # set up initial conditions accounting for restarts
        if not mbar:
            mbar = 0.637712441346 | units.MSun # this is only for 0.1-100 MSun kroupa
        if not mbar_se and time==0 | units.Myr:
            mbar_se = mbar
        if N:
            mass = mbar*N
        else:
            if mass:
                N = mass/mbar
            else:
                raise ValueError('Either N or mass must be set')

        # has to be particles to function with bridge
        particles = Particles(1)
        particles.mass = mass
        particles.position = position
        particles.velocity = velocity

        # set up tidal shock tracking
        self.last_max_evalues = [0,0,0] | units.Gyr**-2
        self.time_of_last_shock = [0,0,0]| units.Myr
        self.eigenvalues = np.empty((0,3)) | units.Gyr**-2

        # storing dt probably good for stability so we don't have big jumps in it
        self.dt = 0.1 | units.Myr


        # if tidal field is present
        super().__init__(N=mass/mbar, mbar = mbar, mbar_se=mbar_se,half_mass_radius=half_mass_radius, kappa=kappa, M_seg=M_seg, n_trhp=n_trhp, particles=particles,
                            grav_instance=grav_instance, stellar_evolution=stellar_evolution, VG=VG, time=time)
    

    def evolve_model(self, tend):
        dt = tend - self.model_time

        # evolve the EMACSS model - leapfrog 
        self.internal_evolution(self.model_time + dt/2)

        # update particle position
        self.particles.position += self.particles.velocity * dt

        # evolve the EMACSS model - leapfrog
        self.internal_evolution(tend)

    def internal_evolution(self, tend):
        # tidal evolution - computes shock mass loss and change of rh due to this
        # self.tidal_shock_evolution(tend)
    
        # relaxation evolution - EMACSS model. updates model time
        self.relaxation_evolution(tend)

        self.particles.mass = self.mbar * self.N

    def tidal_shock_evolution(self, tend):
        dt = tend - self.model_time
        # tidal tensor mass loss as position is not updated in this timestep - make this a funtion
        # Construct the tidal tensor
        eigenvalues = self.tidal_tensor_eigenvalues(4 | units.pc, self.particles.position[0].x, self.particles.position[0].y,
                                                     self.particles.position[0].z)
        
        self.eigenvalues=np.append(self.eigenvalues, eigenvalues, axis=0)

        # assume shock happens evenly across cluster
        index=0
        dN=0
        for lam in eigenvalues:
            # apply the shock for this component if any component drops below 88% of the last maximum and is approximately a minimum
            if np.abs(lam) < 0.88*self.last_max_evalues[index] and np.gradient(np.abs(self.eigenvalues[:,index]))[-1] >= 0:
                # Weinberg coefficients
                Awij = (1 + 0.237 * constants.G * self.N*self.mbar/self.half_mass_radius**3 * (self.model_time-self.time_of_last_shock[-1])**2)**(-3/2)

                # we need to integrate Tij dt over the time since the last shock - use scipy.integrate.simpson
                Itid = np.abs(simpson(self.eigenvalues[int(self.time_of_last_shock[index]/dt):,index],
                                       dx=dt.value_in(units.Gyr))/100)**2 * Awij
                tshock = (self.model_time - self.time_of_last_shock[index]) * 65.6 * (self.particles.mass/(1e4 | units.MSun)) * \
                      (self.half_mass_radius/(4 | units.pc)) ** -3 * Itid ** -1

                dN -= dt*self.N/tshock 
                self.time_of_last_shock[index]=self.model_time
                self.last_max_evalues[index] = lam
            if np.abs(lam) > self.last_max_evalues[index]:
                self.last_max_evalues[index] = np.abs(lam)
            index+=1

        # half mass radius evolution due to tidal shocks
        dr = dN*self.half_mass_radius/self.N*(2-1/tidal_shock_energy_fraction)
        self.N += dN
        self.half_mass_radius += dr

    def relaxation_evolution(self,tend):
        tol = 1e-6
        # rk coefficients
        b21=0.2
        b31=3.0/40.0
        b32=9.0/40.0
        b41=0.3
        b42 = -0.9
        b43=1.2
        b51 = -11.0/54.0 
        b52=2.5
        b53 = -70.0/27.0
        b54=35.0/27.0
        b61=1631.0/55296.0
        b62=175.0/512.0
        b63=575.0/13824.0
        b64=44275.0/110592.0
        b65=253.0/4096.0
        c1=37.0/378.0
        c3=250.0/621.0
        c4=125.0/594.0
        c6=512.0/1771.0
        dc5 = -277.00/14336.0
        dc1=c1-2825.0/27648.0
        dc3=c3-18575.0/48384.0
        dc4=c4-13525.0/55296.0
        dc6=c6-0.25
        # self.dt = tend - self.model_time
        while self.model_time < tend:
            # copy of the initial values
            duplicate_array = self.get_nbody()
            # dt = tend - self.model_time
            while True:
                # we need to solve each step or rk in parallel for each variable so we can update rates accordingly
                self.set_nbody(duplicate_array)

                # ensure we don't overshoot the end time
                self.dt = min(tend - self.model_time, self.dt) 

                # first rk step
                self.trhp = self.relaxation_time_prime()
                dr1 = self.rate_array()
                self.set_nbody(duplicate_array + [self.dt] * (b21 * dr1))
                # second rk step
                self.trhp = self.relaxation_time_prime()
                dr2 = self.rate_array()
                self.set_nbody(duplicate_array + [self.dt] * (b31 * dr1 + b32 * dr2))
                # third rk step
                self.trhp = self.relaxation_time_prime()
                dr3 = self.rate_array() 
                self.set_nbody(duplicate_array + [self.dt] * (b41 * dr1 + b42 * dr2 + b43 * dr3))
                # fourth rk step
                self.trhp = self.relaxation_time_prime()
                dr4 = self.rate_array()
                self.set_nbody(duplicate_array + [self.dt] * (b51 * dr1 + b52 * dr2 + b53 * dr3 + b54 * dr4))
                # fifth rk step
                self.trhp = self.relaxation_time_prime()
                dr5 = self.rate_array()
                self.set_nbody(duplicate_array + [self.dt] * (b61 * dr1 + b62 * dr2 + b63 * dr3 + b64 * dr4 + b65 * dr5))
                # sixth rk step
                self.trhp = self.relaxation_time_prime()
                dr6 = self.rate_array()
                self.set_nbody(duplicate_array + [self.dt] * (c1 * dr1 + c3 * dr3 + c4 * dr4 + c6 * dr6))

                # if (self.model_time+dt > tend): break

                err = max([self.dt] * (dc1 * dr1 + dc3 * dr3 + dc4 * dr4 + dc5 * dr5 + dc6 * dr6)/(tol * self.get_nbody()))
                if err <= 1.0: break
                if self.dt < 1.01 * self.min_step(): break
                step_test = 0.9 * self.dt * (err) ** -0.25
                if self.dt >= (0.0 | units.Myr):
                    self.dt = max(step_test, 0.1 * self.dt)
                else:
                    self.dt = min(step_test, 0.1 * self.dt)

            if (self.dt < self.min_step()): self.dt = self.min_step()
            elif (err >  1.89e-4): self.dt = 0.9*self.dt*err**-0.2
            else: self.dt = 5.0*self.dt

    def get_gravity_at_point(self,radius,x,y,z):
        mass=self.particles.mass[0]
        xx,yy,zz=self.particles[0].position
        
        # possibly set radius to half mass radius
        dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+radius**2)
        
        ax=constants.G*mass*(xx-x)/dr2**1.5
        ay=constants.G*mass*(yy-y)/dr2**1.5
        az=constants.G*mass*(zz-z)/dr2**1.5
        
        return ax,ay,az
    
    # array for storing output
    def output_array(self):
        x, y, z = self.particles[0].position.value_in(units.pc)
        vx, vy, vz = self.particles[0].velocity.value_in(units.km/units.s)
        return np.array([[self.model_time.value_in(units.Myr), x,y,z,vx,vy,vz, self.N, self.mbar.value_in(units.MSun), self.mbar_se.value_in(units.MSun), self.half_mass_radius.value_in(units.pc),self.rtidal().value_in(units.pc), self.n_trhp, self.kappa, self.M_seg]])
    
    def stop(self):
        pass