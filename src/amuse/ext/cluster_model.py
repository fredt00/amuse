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

global xi0
xi0 = 0.0075 #0.0142 for equal mass, 0.0075 for the paper
global zeta
zeta = 0.1#0.1 # reducing this helps match the mass loss profile, but then the half mass radius does not increase enough
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
kappa_1=0.3#0.24 # ours seems to go to 0.9?
# chi between 0 and 1, depends on mass of escaping stars either mlow or mbar
global Chi
Chi= 0.8#0.55#0.35+(1-0.35)*(self.rhalf/rtidal)**1.1#0.3#0.55 in paper, we find for circular orbit we have to set 0.45
global q
q=2 # doesn't seem very sensitive to this
global M1
M1=4
global nc
nc =12.5 # 12.5 in paper ... maybe lowerfor us? 5?
global tidal_shock_energy_fraction
tidal_shock_energy_fraction =0.25 # - could be more physically motivated? written in terms of other parameters?
 
class internal_dynamics(object):
    def __init__(self,N,mbar, rhalf, kappa, M_seg):
        self.N=N
        self.mbar=mbar
        self.rhalf=rhalf
        self.kappa=kappa
        self.M_seg=M_seg
        self.m_low = 0.1 | units.MSun
        self.m_up = 15 | units.MSun
        self.psi=14#8.0#13.5364073081 # this should depend on mass spectrum WITHIN rhalf   - !!!NOTE possibly this should be the case for many parameters including shape parameter and mean mass etc...
        # ok so i think we want mean mass to be for the whole cluster, buy psi should depend only within rhalf... makes it harder to relate to other variables like mbar...
        # perhaps M_seg and or kappa could relate mbar_h to mbar?
        self.rtidal = 0 | units.pc
        self.n_trh = 0
        self.n_trhp = 0
    
    # derived quantities
    def relaxation_time(self):
        return 0.138 * np.sqrt(self.N) * self.rhalf**1.5 / ((constants.G * self.mbar).sqrt() * np.log(gamma_c * self.N))
    def relaxation_time_prime(self):
        return self.relaxation_time()/self.psi
    
    def total_energy(self):
        return -self.kappa * constants.G*(self.N*self.mbar)**2./self.rhalf

    def P(self):
        return ((self.rhalf/self.rtidal)/R1)**z * ((self.N*np.log(gamma_c*1.5e4))/(N1*np.log(gamma_c*self.N)))**(1-0.75)
    
    def F(self):
        F=0.
        if nc/2<=self.n_trhp <= nc:
            F = 2.*self.n_trhp/nc - 1.
        if self.n_trhp > nc:
            F=1.
        return F
    
    # dimensionless rates
    def xi(self):
        return self.xi_i() + self.xi_e()
    def xi_i(self):
        return 0. # stellar evolution 0 for now
    def xi_e(self):
        return self.F()*xi0 * (1-self.P()) + (f + (1-f)*self.F())*3/5 *zeta*self.P()
    
    def gamma(self):
        return self.gamma_dyn() + self.gamma_se()
    def gamma_dyn(self):
        return (1-self.mesc()/self.mbar)* self.S()*self.U()*self.xi()
    def gamma_se(self):
        return 0.

    def lambd(self):
        lambd=0.
        if self.n_trhp > 0.5*nc:
            lambd += (kappa_1 - self.kappa) * (2*self.n_trhp/nc - 1)
        return lambd
    
    def epsilon(self):
        epsilon=zeta
        if self.n_trhp <= nc:
            epsilon = 1./self.kappa * self.mesc()/self.mbar * self.rhalf/self.rtidal * self.xi()
        return epsilon
    
    def mu(self):
        return self.epsilon() -2*self.xi() +2*self.gamma() + self.lambd()
    
    # other derived quantities
    def mesc(self):
        return Chi * (self.mbar - self.m_low) + self.m_low
    
    def S(self):
        return ((self.M_seg-3.)/(M1-3.))**q
    
    def U(self):
        return (self.m_up - self.mbar)/self.m_up
    
    # derived rates divided by variable (so log rate)
    def dNdt(self):
        return - self.xi()/self.relaxation_time_prime()
    def dMsegdt(self):
        return (M1-self.M_seg) /self.relaxation_time_prime()
    def dkdt(self):
        return self.lambd() / self.relaxation_time_prime()
    def drdt(self):
        return self.mu()/self.relaxation_time_prime()
    def dmbardt(self):
        return self.gamma() / self.relaxation_time_prime()
    def dtrhpdt(self):
        # for counting
        return 1./self.relaxation_time_prime()
    
    def dpsidt(self):
        #return -5/2*(1-self.F())*(self.dmbardt()/self.M_seg-self.dMsegdt()/self.M_seg**2)
        # return -5/2*(self.dmbardt() * (self.psi-7.8)-(1-self.F())*self.dMsegdt()/self.M_seg**2)
        # return -5/2 * (self.dmbardt() - 1/3*self.dMsegdt()/self.M_seg)
        return -5/2 *(self.dmbardt() - 0.5*self.dMsegdt())
        # A=0.2 
        # B=2.5
        # return A*self.dMsegdt() -B*self.dmbardt()
    
    def min_step(self):
        return self.relaxation_time_prime()/1e6

class star_cluster_particle(internal_dynamics):
    def __init__(self, mass, half_mass_radius, position, velocity, external_get_gravity_at_point=None):
        # set initial conditions
        self.particles = Particles(1)
        self.particles.mass = mass
        self.particles.position = position
        self.particles.velocity = velocity
        self.model_time = 0 | units.Myr

        # set up tidal shock tracking
        self.tt_last_max = [0,0,0,0,0,0] 
        self.last_max_evalues = [0,0,0]
        self.time_of_last_shock = [0,0,0]| units.Myr
        self.time_of_last_shock_tt = [0,0,0,0,0,0]| units.Myr
        self.tidal_tensor_time = np.empty((0,6))
        self.eigenvalues = np.empty((0,3))
        self.shock_times = [] | units.Myr
        self.shock_times_types = []


        super().__init__(N=18015, mbar=0.555131467864 | units.MSun, rhalf=half_mass_radius, kappa=0.2, M_seg=3)

        
        self.dm_shock= 0 | units.MSun
 
        self.external_get_gravity_at_point = external_get_gravity_at_point

    def evolve_model(self, tend):
        dt = tend - self.model_time

        # evolve the EMACSS model - leapfrog 
        self.internal_evolution(self.model_time+dt/2)

        # update particle position
        self.particles.position+=self.particles.velocity*dt

        # evolve the EMACSS model - leapfrog
        self.internal_evolution(tend)

    def internal_evolution(self, tend):
        # think about the order here!
        # tidal evolution - computes shock mass loss and change of rh, returns current rtidal
        self.tidal_evolution(tend)
    
        # relaxation evolution 
        self.relaxation_evolution(tend)

        self.particles.mass = self.mbar * self.N
        self.model_time=tend

    def tidal_evolution(self, tend):
        dt = tend - self.model_time
        # here we do the same rk steps for all quantities, then check error with N as this is most important
        # tidal tensor mass loss as position is not updated in this timestep - make this a funtion
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
        self.rtidal = (constants.G * self.N*self.mbar/T)**(1/3)
        print(self.rtidal.in_(units.pc))
        self.dm_shock = 0 | units.MSun
        index=0
        for lam in eigenvalues:
            if np.abs(lam) < 0.88*self.last_max_evalues[index] and np.gradient(np.abs(self.eigenvalues[:,index]))[-1] >= 0:
                # apply the shock for this component if any component drops below 88% of the last maximum and is approximately a minimum
                # Awij = (1 + 0.237 * constants.G * self.particles.mass[0]/self.rhalf[0]**3 * (self.model_time-self.time_of_last_shock[-1])**2)**(-3/2)
                Awij = 1
                # we need to integrate Tij dt over the time since the last shock - use scipy.integrate.simpson
                Itid = np.abs(simpson(self.eigenvalues[int(self.time_of_last_shock[index]/dt):,index], dx=dt.value_in(units.Gyr))/100)**2 * Awij
                tshock = (self.model_time - self.time_of_last_shock[index]) * 65.6 * (self.particles.mass/(1e4 | units.MSun)) * (self.rhalf/(4 | units.pc))**-3\
                * (Itid)**-1
                # we can experiment here perhaps with form of mass loss - will require experimentation
                self.N -=dt*self.N/tshock # interestingly we seem to have made this more complicated than the paper!
                self.time_of_last_shock[index]=self.model_time
                self.last_max_evalues[index] = lam
            else:
                self.Itid = 0 
            if np.abs(lam) > self.last_max_evalues[index]:
                self.last_max_evalues[index] = np.abs(lam)
            index+=1

        # half mass radius evolution due to tidal shocks
        # self.rhalf +=self.rhalf/(self.N*self.mbar)*self.dm_shock*(2-1/tidal_shock_energy_fraction)
    
    def relaxation_evolution(self, tend):
        # try a small initial timestep - eventually this should only happen for very first call
        internal_time = self.model_time
        tol = 1e-6
        dt = tend - internal_time
        while internal_time < tend:
            while True:
                # update eqns 7-11 via adaptive 5th order RK
                # order to solve in is N, mbar, rhalf, kappa, M_seg
                N_5, N_err = self.rk5_err(self.dNdt(),internal_time,self.N,dt)
                mbar_5, mbar_err = self.rk5_err(self.dmbardt(),internal_time,self.mbar,dt)
                rhalf_5, rhalf_err = self.rk5_err(self.drdt(),internal_time,self.rhalf,dt)
                kappa_5, kappa_err = self.rk5_err(self.dkdt(),internal_time,self.kappa,dt)
                Mseg_5, Mseg_err = self.rk5_err(self.dMsegdt(),internal_time,self.M_seg,dt)
                psi_5, psi_err = self.rk5_err(self.dpsidt(),internal_time,self.psi,dt)
                ##^^ make grad zero after certain number of nc!!!!!!
                err = max(N_err, mbar_err, rhalf_err, kappa_err, Mseg_err, psi_err)/tol# maybe this should be energy error? or the max error in the 4 quantities
                if err <= 1.0: break
                if dt < 1.01*self.min_step(): break
                step_test = 0.9*dt*(err)**-0.25
                if dt>=(0.0 | units.Myr):
                    dt= max(step_test,0.1*dt)
                else:
                    dt=min(step_test,0.1*dt)
            self.n_trh += dt/self.relaxation_time()
            self.n_trhp += dt/self.relaxation_time_prime()

            self.N = N_5
            self.mbar = mbar_5
            self.rhalf = rhalf_5
            self.kappa = kappa_5
            self.M_seg = Mseg_5
            self.psi = psi_5
            internal_time += dt
        print(internal_time.in_(units.Myr))
    def get_tidalfield_at_point_per_gyr_sq(self,scale, x, y, z):
        # perhaps this could vary, = self.rhalf
        h = scale
        ax0,ay0,az0 = self.external_get_gravity_at_point(0 | units.pc, x, y, z)
        axx,ayx,azx = self.external_get_gravity_at_point(0 | units.pc, x+h, y, z)
        axy,ayy,azy = self.external_get_gravity_at_point(0 | units.pc, x, y+h, z)
        axz,ayz,azz = self.external_get_gravity_at_point(0 | units.pc, x, y, z+h)
        Txx = ((axx-ax0)/h).value_in(units.gyr**-2)
        Tyy = ((ayy-ay0)/h).value_in(units.gyr**-2)
        Tzz = ((azz-az0)/h).value_in(units.gyr**-2)
        Txy = ((axy-ax0)/h).value_in(units.gyr**-2)
        Txz = ((axz-ax0)/h).value_in(units.gyr**-2)
        Tyz = ((ayz-ay0)/h).value_in(units.gyr**-2)

        return Txx, Tyy, Tzz, Txy, Txz, Tyz
    
    def get_gravity_at_point(self,radius,x,y,z):
        mass=self.particles.mass[0]
        xx,yy,zz=self.particles[0].position
        
        # possibly set radius to half mass radius
        dr2=((xx-x)**2+(yy-y)**2+(zz-z)**2+radius**2)
        
        ax=constants.G*mass*(xx-x)/dr2**1.5
        ay=constants.G*mass*(yy-y)/dr2**1.5
        az=constants.G*mass*(zz-z)/dr2**1.5
        
        return ax,ay,az
    
    def rk5_err(self, f, x, y, h):
        k1 = h*f*y
        k2 = h*f*(y+((1/5)*k1)) 
        k3 = h*f*(y+((3/40)*k1)+((9/40)*k2))
        k4 = h*f*(y+((3/10)*k1)-((9/10)*k2)+((6/5)*k3))
        k5 = h*f*(y-((11/54)*k1)+((5/2)*k2)-((70/27)*k3)+((35/27)*k4))
        k6 = h*f*(y+((1631/55296)*k1)+((175/512)*k2)+((575/13824)*k3)+((44275/110592)*k4)+((253/4096)*k5))
        yn4 = y + ((37/378)*k1)+((250/621)*k3)+((125/594)*k4)+((512/1771)*k6)
        yn5 = y + ((2825/27648)*k1)+((18575/48384)*k3)+((13525/55296)*k4)+((277/14336)*k5)+((1/4)*k6)
        err = np.abs((yn4-yn5)/yn4)

        truncation_error = np.abs((k1/150. - 3./100.*k3 + 16./75.*k4 + 1./20.*k5 + 6./25.*k6)/yn5)
        return yn4, err