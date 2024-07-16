#####################################
# Cluster model particle for use in AMUSE
#####################################
from amuse.units import constants
from amuse.datamodel import Particles
from amuse.units import units
import numpy as np
from scipy.integrate import simpson
from amuse.ic.brokenimf import MultiplePartIMF

# Define fixed parameters
xi0 = 0.0075 #0.0142 for equal mass, 0.0075 for the paper
zeta = 0.1
N1 = 1000 # 15000 and for equal mass clusters
R1 =0.22
gamma_c = 0.02 # 0.11 for equal mass clusters
psi1 = 9.46793200131#8
f=0.3
z=2#2 #1.61 for equal mass clusters
kappa_1=0.24 # ours seems to go to 0.9?
# chi between 0 and 1, depends on mass of escaping stars either mlow or mbar
Chi = 0.3#0.35+(1-0.35)*(self.rhalf/rtidal)**1.1#0.3#0.55 in paper, we find for circular orbit we have to set 0.45
q=2
M1=4
nc =5#10 # 12.5 in paper ... maybe lowerfor us? 5?
# compute F, requires Rch first - this should account for core collapse? (from emacss paper)
# N2 = 12
# N3 = 15000
# Rch = (N2/N + N2/N3)**(2/3)
# Rch_min =  0.0086177# check this!1
# F = Rch_min/Rch

class star_cluster_particle(object):
    def __init__(self, mass, half_mass_radius, position, velocity, get_gravity_at_point=None):
        # set initial conditions
        self.particles = Particles(1)
        self.particles.mass = mass
        self.particles.position = position
        self.particles.velocity = velocity
        self.initial_mass = mass
        self.rhalf = half_mass_radius
        self.model_time = 0 | units.Myr

        # set up tidal shock tracking
        self.tt_last_max = [0,0,0,0,0,0] 
        self.last_max_evalues = [0,0,0]
        self.time_of_last_shock = [0,0,0]| units.Myr
        self.time_of_last_shock_tt = [0,0,0,0,0,0]| units.Myr
        self.Itid = 0 
        self.tidal_tensor_time = np.empty((0,6))
        self.eigenvalues = np.empty((0,3))
        self.shock_times = [] | units.Myr
        self.shock_times_types = []

        # set up evolving cluster model parameters
        self.mbar = 0.547 | units.MSun #0.515
        self.n_trh = 0
        self.M_seg = 3
        self.kappa = 0.201199700706#0.2 # check this!
        self.psi = 8#13.5364073081
        self.m_low = 0.1 | units.MSun
        self.m_up = 15 | units.MSun
        self.IMF =  MultiplePartIMF(
            mass_boundaries=[0.01, 0.08, 0.5, 100.0] | units.MSun,
            mass_min=self.m_low,
            mass_max=self.m_up,
            alphas=[-0.3, -1.3, -2.3],
        )
        self.dm_shock= 0 | units.MSun
 
        self.get_gravity_at_point = get_gravity_at_point

    def evolve_model(self, tend):
        dt = tend - self.model_time

        # evolve the EMACSS model - leapfrog 
        self.internal_evolution(self.model_time+dt/2)

        # update particle position
        self.particles.position+=self.particles.velocity*dt

        # evolve the EMACSS model - leapfrog
        self.internal_evolution(tend)

    def internal_evolution(self, tend):
        dt = tend - self.model_time

        # Construct the tidal tensor
        Txx, Tyy, Tzz, Txy, Txz, Tyz = self.get_tidalfield_at_point_per_gyr_sq(4 | units.pc, self.particles.position.x[0], self.particles.position.y[0], self.particles.position.z[0])
        tidal_tensor = np.array([[Txx, Txy, Txz],
                                 [Txy, Tyy, Tyz],
                                 [Txz, Tyz, Tzz]])
        self.tidal_tensor_time=np.append(self.tidal_tensor_time,np.array([[Txx, Tyy, Tzz, Txy, Txz, Tyz]]),axis=0)

        # tidal_tensor is a 3x3 matrix, write code that computes the maximum eigenvalue
        eigenvalues, _ = np.linalg.eig(tidal_tensor)
        self.eigenvalues = np.append(self.eigenvalues, np.array([eigenvalues]), axis=0)
        max_eigenvalue = np.max(np.abs(eigenvalues))| units.gyr**-2
        omegasq = (np.abs(eigenvalues.sum())/3) | units.gyr**-2
        T = max_eigenvalue + omegasq
        rtidal = (constants.G * self.particles.mass/T)**(1/3)

        # relaxation mass loss - this accounts for the background tidal field!
        N = self.particles.mass / self.mbar
        m_esc = Chi * (self.mbar - self.m_low) + self.m_low
        trh = 0.138 * np.sqrt(N) * self.rhalf**1.5 / ((constants.G * self.mbar).sqrt() * np.log(gamma_c * N))
        # self.psi += dt * (psi1 - self.psi) / ()
        # self.psi = (3.1-(N*m_esc.value_in(units.MSun))**(5/2))/self.mbar.value_in(units.MSun)**(5/2)  # could submit mean escape mass from numerator!
        trh/=self.psi
       
        self.n_trh += dt/trh
        if self.n_trh <nc/2:
            lambd = 0
            F=0
        elif nc/2<=self.n_trh <= nc:
            F = 2*self.n_trh/nc - 1
            lambd = (kappa_1 - self.kappa)* F
        else:
            F=1
            lambd = (kappa_1 - self.kappa)* (2*self.n_trh/nc - 1)
            
        P = ((self.rhalf/rtidal)/R1)**z * ((N*np.log10(gamma_c*1.5e4))/(N1*np.log10(gamma_c*N)))**(1-0.75)
        xi = F*xi0 * (1-P) + (f + (1-f)*F)*3/5 *zeta*P

        if self.n_trh <= nc:
            epsilon = 1/self.kappa * m_esc/self.mbar * self.rhalf/rtidal * xi
        else:
            epsilon = zeta
    
        # mass loss is number time mean escaper mass
        dN = -N* xi *dt/trh
        dm_relax= dN * m_esc
        
        # update the cluster parameters
        self.M_seg += self.M_seg * (M1-self.M_seg)*dt /(trh) # was 1 before!
        S = ((self.M_seg-3)/(M1-3))**q
        U = (self.m_up - self.mbar)/self.m_up
        gamma_e = (1-m_esc/self.mbar)* S*U*xi
        gamma_s = 0 # no stellar evolution for now
        gamma = gamma_s + gamma_e
        # evolve mbar due to escape of stars and stellar evolution
        self.mbar += gamma *self.mbar*dt/trh
        # here we could simply make mlow(t) as well as mup(t) (once SE is on) - then for an IMF this is analytic
        # was 3.1
        # redefine IMF with new mbar given that m_up is set by SE and m_low is set by escapers
        # set m_low by integrating IMF up until mass=dm_relax 
        # or should we find the IMF that has Mbar?
        # self.m_low+= 0.1*gamma * self.m_low * dt/trh
        # self.IMF =  MultiplePartIMF(
        #     mass_boundaries=[0.01, 0.08, 0.5, 100.0] | units.MSun,
        #     mass_min=self.m_low,
        #     mass_max=self.m_up,
        #     alphas=[-0.3, -1.3, -2.3],
        # )
        #self.psi = self.IMF.mass2p5_mean()/self.mbar**(5/2)
        # self.psi -= 5/2 * self.psi * gamma*dt/trh
        # self.psi += 1.4*5/2*gamma*dt * (psi1 - self.psi) / (5 | units.Gyr)#trh

        self.kappa += lambd * self.kappa * dt/trh


        # shock mass loss - this accounts for the tidal tensor
        self.dm_shock = 0 | units.MSun
        index=0
        for lam in eigenvalues:
            if np.abs(lam) < 0.88*self.last_max_evalues[index] and np.gradient(np.abs(self.eigenvalues[:,index]))[-1] >= 0:
                # apply the shock for this component if any component drops below 88% of the last maximum and is approximately a minimum
                # Awij = (1 + 0.237 * constants.G * self.particles.mass[0]/self.rhalf[0]**3 * (self.model_time-self.time_of_last_shock[-1])**2)**(-3/2)
                Awij = 1
                # we need to integrate Tij dt over the time since the last shock - use scipy.integrate.simpson
                self.Itid = np.abs(simpson(self.eigenvalues[int(self.time_of_last_shock[index]/dt):,index], dx=dt.value_in(units.Gyr))/100)**2 * Awij
                tshock = (self.model_time - self.time_of_last_shock[index]) * 65.6 * (self.particles.mass/(1e4 | units.MSun)) * (self.rhalf/(4 | units.pc))**-3\
                * (self.Itid)**-1
                self.dm_shock -=dt*N*m_esc/tshock
                self.time_of_last_shock[index]=self.model_time
                self.last_max_evalues[index] = lam
            else:
                self.Itid = 0 
            if np.abs(lam) > self.last_max_evalues[index]:
                self.last_max_evalues[index] = np.abs(lam)
            index+=1

        # half mass radius evolution due to relaxation
        mu = epsilon -2*xi +2*gamma + lambd
        self.rhalf += mu*self.rhalf * dt/trh

        # the raina campos rhalf evolution
        # self.rhalf += self.rhalf/self.particles.mass*((2-zeta/xi)* dm_relax + dm_shock*(2-1/f))

        # half mass radius evolution due to tidal shocks
        f =0.25 # - could be more physically motivated? written in terms of other parameters?
        self.rhalf +=self.rhalf/self.particles.mass*self.dm_shock*(2-1/f)
        self.particles.mass += dm_relax + self.dm_shock

        self.model_time= tend

        def get_tidalfield_at_point_per_gyr_sq(scale, x, y, z):
            # perhaps this could vary, = self.rhalf
            h = scale
            ax0,ay0,az0 = self.get_gravity_at_point(0 | units.pc, x, y, z)
            axx,ayx,azx = self.get_gravity_at_point(0 | units.pc, x+h, y, z)
            axy,ayy,azy = self.get_gravity_at_point(0 | units.pc, x, y+h, z)
            axz,ayz,azz = self.get_gravity_at_point(0 | units.pc, x, y, z+h)
            Txx = ((axx-ax0)/h).value_in(units.gyr**-2)
            Tyy = ((ayy-ay0)/h).value_in(units.gyr**-2)
            Tzz = ((azz-az0)/h).value_in(units.gyr**-2)
            Txy = ((axy-ax0)/h).value_in(units.gyr**-2)
            Txz = ((axz-ax0)/h).value_in(units.gyr**-2)
            Tyz = ((ayz-ay0)/h).value_in(units.gyr**-2)

            return Txx, Tyy, Tzz, Txy, Txz, Tyz