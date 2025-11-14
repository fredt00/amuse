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
from amuse.datamodel.particle_attributes import HopContainer
import os, re, yt
from yt.utilities.cosmology import Cosmology
import glob
from yt.units import Myr, Gyr, km, s, Mpc  
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
    def __init__(self,grav_instance=None, yt_file_path=None, sink_id=None):
        self.grav_instance=grav_instance
        if yt_file_path: self.parse_tidal_data_fine(yt_file_path, sink_id)
        else: self.sink_info=None
                
    def parse_tidal_data_fine(self, file_path, sink_id):
        """
        Read log_TT/TT_{sink_id:05d}.dat and store all arrays as AMUSE quantities.

        Assumptions:
        - Columns: [t_code, aexp, x, y, z, Txx, Tyy, Tzz, Txy, Txz, Tyz]
        - Positions (x,y,z) are in RAMSES code length units (comoving).
        - Tidal tensor components are in RAMSES code units of 1/(code_time)^2.

        Stored fields (all AMUSE quantities):
        t_code (Myr), aexp (dimensionless), z (dimensionless),
        cosmic_time (Gyr), lookback_time (Gyr), since_form (Gyr),
        x,y,z (kpc, proper), Txx..Tyz (gyr^-2), plus a few bookkeeping params.
        """

        # ---- locate and load one RAMSES output to get units/cosmology
        info_candidates = sorted(glob.glob(os.path.join(file_path, "output_*", "info_*.txt")))
        if not info_candidates:
            raise FileNotFoundError(f"No RAMSES info_*.txt under {file_path}/output_*/")
        ds = yt.load(info_candidates[0])

        # Base code units (authoritative from info_*.txt)
        unit_t = ds.quan(ds.parameters["unit_t"], "s")   # code time unit
        unit_l = ds.quan(ds.parameters["unit_l"], "cm")  # code length unit
        a0 = float(ds.parameters.get("aexp", 1.0))       # scale factor for this output

        # Build a yt Cosmology when this is a cosmological run
        is_cosmo = bool(getattr(ds, "cosmological_simulation", False))
        if is_cosmo:
            # h, Omega_m, Omega_L are available on ds for RAMSES
            h = float(getattr(ds, "hubble_constant", 0.0))
            omegam = float(getattr(ds, "omega_matter", 0.0))
            omegal = float(getattr(ds, "omega_lambda", 0.0))
            cosmo = Cosmology(hubble_constant=h, omega_matter=omegam, omega_lambda=omegal)
            # optional: H0 in 1/Gyr
            H0_1_over_gyr = (h * 100.0 * km/s/Mpc).to(1/Gyr).value | (1/units.gyr)
        else:
            h = omegam = omegal = 0.0
            cosmo = None
            H0_1_over_gyr = 0.0 | (1/units.gyr)

        # ---- read the per-sink TT file
        tt_dir = os.path.join(file_path, "log_TT")
        tt_path = os.path.join(tt_dir, f"TT_{int(sink_id):05d}.dat")
        if not os.path.isfile(tt_path):
            raise FileNotFoundError(f"Missing TT file: {tt_path}")

        data = np.genfromtxt(tt_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Columns
        t_code = data[:, 0]      # code time (dimensionless multiplier of unit_t)
        aexp   = data[:, 1]
        x_cu   = data[:, 2]
        y_cu   = data[:, 3]
        z_cu   = data[:, 4]
        Txx_cu = data[:, 5]
        Tyy_cu = data[:, 6]
        Tzz_cu = data[:, 7]
        Txy_cu = data[:, 8]
        Txz_cu = data[:, 9]
        Tyz_cu = data[:,10]

        # ---- times
        # Code time as AMUSE quantity (in Myr for convenience)
        t_code_myr_vals = (t_code * unit_t).to(Myr).value
        t_code = t_code_myr_vals | units.Myr

        if is_cosmo:
            z_vals = 1.0 / aexp - 1.0
            lookback_gyr_vals    = cosmo.lookback_time(z_vals).to(Gyr).value
            cosmic_time_gyr_vals = cosmo.t_from_z(z_vals).to(Gyr).value
            z_q   = z_vals | units.none
            aexp_q = aexp  | units.none
            lookback_time = lookback_gyr_vals | units.gyr
            cosmic_time   = cosmic_time_gyr_vals | units.gyr
        else:
            # Non-cosmological: treat code time as the physical clock; aexp≈1, z≈0
            z_q    = np.zeros_like(aexp) | units.none
            aexp_q = aexp | units.none
            lookback_time = np.zeros_like(aexp) | units.gyr
            cosmic_time   = t_code.as_quantity_in(units.gyr)

        # since_form: ensure a forward-running clock from the earliest entry
        since_form = cosmic_time - cosmic_time.min()

        # ---- positions: code length -> proper kpc (multiply by a/a0)
        x_kpc_vals = (x_cu * unit_l * (aexp / a0)).to("kpc").value
        y_kpc_vals = (y_cu * unit_l * (aexp / a0)).to("kpc").value
        z_kpc_vals = (z_cu * unit_l * (aexp / a0)).to("kpc").value
        x = x_kpc_vals | units.kpc
        y = y_kpc_vals | units.kpc
        z = z_kpc_vals | units.kpc

        # ---- tidal tensor: code units (1/unit_t^2) -> Gyr^-2, stored as AMUSE quantities
        inv_t_gyr = (1.0 / unit_t).to(1/Gyr).value  # numeric in 1/Gyr
        fac = inv_t_gyr**2
        Txx = (Txx_cu * fac) | (units.gyr**-2)
        Tyy = (Tyy_cu * fac) | (units.gyr**-2)
        Tzz = (Tzz_cu * fac) | (units.gyr**-2)
        Txy = (Txy_cu * fac) | (units.gyr**-2)
        Txz = (Txz_cu * fac) | (units.gyr**-2)
        Tyz = (Tyz_cu * fac) | (units.gyr**-2)

        # ---- package everything as AMUSE quantities
        sink = dict(
            t_code=t_code,                    # Myr
            aexp=aexp_q, z=z_q,               # dimensionless
            lookback_time=lookback_time,      # Gyr
            cosmic_time=cosmic_time,          # Gyr
            since_form=since_form,            # Gyr (starts at 0)
            x=x, y=y, z=z,                    # kpc (proper)
            Txx=Txx, Tyy=Tyy, Tzz=Tzz, Txy=Txy, Txz=Txz, Tyz=Tyz,   # gyr^-2

            # bookkeeping (all AMUSE quantities as requested)
            unit_t= (unit_t.to("s").value | units.s),
            unit_l= (unit_l.to("cm").value | units.cm),
            a0= (a0 | units.none),
            h= (h | units.none),
            omega_m=(omegam | units.none),
            omega_l=(omegal | units.none),
            H0=H0_1_over_gyr,   # 1/gyr; 0 if non-cosmological
        )

        # ---- deduplicate by cosmic time (keep LAST occurrence)
        # Use float-valued keys to handle AMUSE quantities.
        ct_vals = sink["cosmic_time"].value_in(units.gyr)
        # Keep last index for each time (round to reduce floating-point noise)
        keys = np.round(ct_vals, decimals=12)
        last_index = {}
        for i, tval in enumerate(keys):
            last_index[tval] = i
        keep = np.array(sorted(last_index.values()))

        for k, v in list(sink.items()):
            # Each v is an AMUSE quantity array or scalar; index arrays only
            try:
                sink[k] = v[keep]
            except Exception:
                sink[k] = v  # scalars: leave as-is

        self.sink_info = sink

    def get_tidal_field_at_time_proper_time(self, model_time):
        """
        Interpolate the tidal tensor at a given model_time measured on the
        'since_form' clock (same definition as self.sink_info['since_form']).

        Parameters
        ----------
        model_time : AMUSE quantity
            Time since formation (e.g. 0.25 | units.gyr).

        Returns
        -------
        Txx, Tyy, Tzz, Txy, Txz, Tyz : AMUSE quantities (units.gyr**-2)

        Notes
        -----
        - If model_time lies outside the tabulated range, the result is clamped
        to the nearest edge (numpy.interp behavior).
        - Works with scalar or array-like model_time.
        """

        if not hasattr(self, "sink_info") or "since_form" not in self.sink_info:
            raise RuntimeError("sink_info with 'since_form' is not available. "
                            "Run parse_tidal_data_fine(...) first.")

        # 1) Time axis (since_form) in Gyr, sorted
        t_sf = self.sink_info["since_form"]              # AMUSE quantity
        t_gyr = np.asarray(t_sf.value_in(units.gyr))     # float array
        order = np.argsort(t_gyr)
        t_sorted = t_gyr[order]

        # 2) Helper: interpolate one component in gyr^-2 and return AMUSE quantity
        def interp_comp(key):
            arr = self.sink_info[key]                                    # AMUSE quantity
            y_sorted = np.asarray(arr.value_in(units.gyr**-2))[order]    # floats
            xq = np.asarray(model_time.value_in(units.gyr))              # floats (scalar or array)
            yq = np.interp(xq, t_sorted, y_sorted)                       # floats
            return yq | (units.gyr**-2)

        # 3) Interpolate all six components
        Txx = interp_comp("Txx")
        Tyy = interp_comp("Tyy")
        Tzz = interp_comp("Tzz")
        Txy = interp_comp("Txy")
        Txz = interp_comp("Txz")
        Tyz = interp_comp("Tyz")

        return Txx, Tyy, Tzz, Txy, Txz, Tyz
        
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
        max_eigenvalue = eigenvalues.max() # note we had an abs in here before which was an error
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
    
    # new routines using potential
    def get_tidalfield_at_point_pot(self,scale,x,y,z):
        h = scale
        # on axis terms
        phi_x0 = self.grav_instance.get_potential_at_point(0 | units.pc, x, y, z)
        phi_px = self.grav_instance.get_potential_at_point(0 | units.pc, x+h, y, z)
        phi_mx = self.grav_instance.get_potential_at_point(0 | units.pc, x-h, y, z)
        phi_py = self.grav_instance.get_potential_at_point(0 | units.pc, x, y+h, z)
        phi_my = self.grav_instance.get_potential_at_point(0 | units.pc, x, y-h, z)
        phi_pz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y, z+h)
        phi_mz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y, z-h)

        Txx = -(phi_px-2*phi_x0+phi_mx)/h**2
        Tyy = -(phi_py-2*phi_x0+phi_my)/h**2
        Tzz = -(phi_pz-2*phi_x0+phi_mz)/h**2

        # off axis terms
        phi_pxpy = self.grav_instance.get_potential_at_point(0 | units.pc, x+h, y+h, z)
        phi_mxmy = self.grav_instance.get_potential_at_point(0 | units.pc, x-h, y-h, z)
        phi_pxmy = self.grav_instance.get_potential_at_point(0 | units.pc, x+h, y-h, z)
        phi_mxpy = self.grav_instance.get_potential_at_point(0 | units.pc, x-h, y+h, z)

        phi_pxpz = self.grav_instance.get_potential_at_point(0 | units.pc, x+h, y, z+h)
        phi_mxmz = self.grav_instance.get_potential_at_point(0 | units.pc, x-h, y, z-h)
        phi_pxmz = self.grav_instance.get_potential_at_point(0 | units.pc, x+h, y, z-h)
        phi_mxpz = self.grav_instance.get_potential_at_point(0 | units.pc, x-h, y, z+h)

        phi_pypz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y+h, z+h)
        phi_mymz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y-h, z-h)
        phi_pymz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y+h, z-h)
        phi_mypz = self.grav_instance.get_potential_at_point(0 | units.pc, x, y-h, z+h)

        Txy = -(phi_pxpy + phi_mxmy - phi_pxmy - phi_mxpy)/(4*h**2)
        Txz = -(phi_pxpz + phi_mxmz - phi_pxmz - phi_mxpz)/(4*h**2)
        Tyz = -(phi_pypz + phi_mymz - phi_pymz - phi_mypz)/(4*h**2)
        return Txx,Tyy,Tzz,Txy,Txz,Tyz
    
    def get_tidalfield_at_point_pot_per_gyr_sq(self,scale,x,y,z):
        h = scale
        Txx,Tyy,Tzz,Txy,Txz,Tyz=self.get_tidalfield_at_point_pot(h,x,y,z)
        Txx=Txx.value_in(units.gyr**-2)
        Tyy=Tyy.value_in(units.gyr**-2)
        Tzz=Tzz.value_in(units.gyr**-2)
        Txy=Txy.value_in(units.gyr**-2)
        Txz=Txz.value_in(units.gyr**-2)
        Tyz=Tyz.value_in(units.gyr**-2)
        return Txx,Tyy,Tzz,Txy,Txz,Tyz
    
    def tidal_tensor_eigenvalues_pot(self, scale, x, y, z):
        Txx, Tyy, Tzz, Txy, Txz, Tyz = self.get_tidalfield_at_point_pot_per_gyr_sq(scale, x, y, z)
        tidal_tensor = np.array([[Txx, Txy, Txz],
                                [Txy, Tyy, Tyz],
                                [Txz, Tyz, Tzz]])
        eigenvalues, _ = np.linalg.eig(tidal_tensor)
        eigenvalues = eigenvalues | units.gyr**-2
        return eigenvalues
    
    def tidal_radius_pot(self, scale, x, y, z, satellite_mass):
        eigenvalues = self.tidal_tensor_eigenvalues_pot(scale, x, y, z)
        max_eigenvalue = eigenvalues.max() # error here!! this should just be np.max(evalues)
        omegasq = np.abs(eigenvalues.sum())/3
        T = max_eigenvalue + omegasq
        return (constants.G * satellite_mass/T)**(1/3)

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
  
        # evolve to 0 Myr so we have dt_soft and other parameters set
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
            self.f2s = self.particles.new_channel_to(self.stellar_evolution.particles, attributes=["mass", "radius", "x", "y", "z", "vx", "vy", "vz", "unbound_time"])
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
            # while dt<(tend-self.bound.model_time):

            #     # compute the integer timestep just below the minimum SE timestep
            #     dt_se_nbody = self.converter.to_nbody(dt).number
            #     n_min_se_time_step = math.ceil(math.log(dt_se_nbody, 0.5))
            #     if n_min_se_time_step > maximum_n_allowed:
            #         n_min_se_time_step = maximum_n_allowed
            #     dt = self.converter.to_si(0.5**n_min_se_time_step | nbody_system.time)

            #     self.f2s.copy()
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

            #     self.f2s.copy()
            #     self.stellar_evolution.evolve_model(self.bound.model_time)
            #     self.s2f.copy()

            #     dt = self.stellar_evolution.particles.time_step.min()

            # # evolve by whatever time is left in the bridge timestep
            # remaining_time = tend-self.bound.model_time

            # self.f2s.copy()
            # self.stellar_evolution.evolve_model(self.bound.model_time+remaining_time/2)
            # self.s2f.copy()
   
            # self.bound.parameters.dt_soft = self.converter.to_si(0.5**maximum_n_for_dt_soft | nbody_system.time)
            # self.f2b.copy()
            # self.bound.evolve_model(self.bound.model_time+remaining_time)
            # self.b2f.copy()

            # self.f2s.copy()
            # self.stellar_evolution.evolve_model(tend)
            # self.s2f.copy()

            # # reset the tree timestep to autodetermination
            # self.bound.parameters.dt_soft=0 | units.Myr
            # self.bound.evolve_model(self.bound.model_time)

        else:

            self.f2b.copy()
            self.bound.evolve_model(tend)
            self.b2f.copy()

        self.f2u.copy()
        self.unbound._evolve_model(tend) # update the unbound particles - this should work ok in bridge because evolves happen after kicking
        self.u2f.copy()
        self.model_time = tend

    def transfer_unbound_particles(self):
        ######### NEW ATTEMPT
        # This must be recursive - we compute the energy of all particles outside the tidal radius and remove the highest +ve energy one - then recompute energy of all outside the new tidal radius etc. repeat until no particles are removed
        # while True:
        #     core = self.bound.particles.cluster_core(self.converter, density_weighting_power=2, reuse_hop=False, hop=HopContainer())
        #     position=self.bound.particles.position-core.position
        #     r2=position.lengths_squared()

        #     # find the particles outside the tidal radius - only compute energy of these
        #     tidal_radius = self.tidal_radius(4|units.pc, core.position.x, core.position.y, core.position.z, self.bound.particles.total_mass())
        #     outside = self.bound.particles[r2 > tidal_radius**2]
        #     if len(outside) == 0:
        #         break

        #     # compute total energies of these particles 
        #     energies = 0.5*(outside.velocity-core.velocity).lengths()**2 + self.bound.get_potential(outside.index_in_code)
        #     a_max = energies.argmax()
        #     # remove the particle with the highest positive energy - if all negative then break
        #     if energies[a_max] > 0 | units.erg/units.kg:
        #         # update the unbound particles
        #         to_remove = outside[a_max]
        #         self.particles[self.particles.key==to_remove.key].unbound_time = self.model_time
        #         self.unbound.particles.add_particle(to_remove)
        #         self.bound.particles.remove_particle(to_remove)
        #     else:
        #         break
        
        # for now lets do what petar does which is remove all particles outside 20rh - quick and conservative
        core = self.bound.particles.cluster_core(self.converter, density_weighting_power=2, reuse_hop=False, hop=HopContainer())
        position=self.bound.particles.position-core.position
        r2=position.lengths_squared()
        # tidal_radius = 20*self.half_mass_radius()
        tidal_radius = 400 | units.pc # try to remove extreme escapers for now
        # print(tidal_radius.in_(units.pc))
        new_outside = self.bound.particles[r2 > tidal_radius**2]#.difference(self.unbound.particles).copy()
        # new_inside = self.particles[r2 <= tidal_radius**2]#.difference(self.bound.particles).copy()
        # print(new_outside)
        # print(new_inside)
        # self.bound.particles.add_particles(new_inside)
        self.bound.particles.remove_particles(new_outside)
        # self.unbound.particles.add_particles(new_outside)
        self.unbound.particles.remove_particles(new_outside)

        # redefine channels
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
