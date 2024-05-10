from amuse.ext.dynamical_friction import dynamical_friction, NFW_profile
from amuse.lab import *
import amuse.couple.bridge as bridge
from amuse.datamodel import Particles
import matplotlib.pyplot as plt
from amuse.plot import plot
from amuse.community.petar.interface import petar

halo_model = NFW_profile(4586423.83576 | units.MSun/units.kpc**3, 4.40907140716 | units.kpc)
R0 = 4.43 | units.kpc
dt = 1 |units.Myr
vy = halo_model.circular_velocity(R0)
# test_mass = Particles(1)
# test_mass.mass = 1e7 | units.MSun
# test_mass.radius = 1 | units.RSun
# test_mass.position = [R0.value_in(units.kpc),0,0] | units.kpc
# test_mass.velocity = [0, vy.value_in(units.kms), 0] | units.kms 

conv = converter = nbody_system.nbody_to_si(1e6 | units.MSun, 17 | units.pc)
test_mass=new_plummer_model(number_of_particles=10000,convert_nbody=conv)
test_mass.position +=[R0.value_in(units.kpc),0,0] | units.kpc
test_mass.velocity +=[0,vy.value_in(units.kms),0] | units.kms
test_mass.mass = 100 | units.MSun
converter = nbody_system.nbody_to_si(test_mass.total_mass(), dt)
print(test_mass.LagrangianRadii(mf=[0.5])[0][0].in_(units.pc))

# gravity = BHTree(converter)
# gravity.parameters.timestep = .5 | nbody_system.time
gravity = petar(converter,mode='openmp')
gravity.particles.add_particles(test_mass)

df_model = dynamical_friction(halo_model, gravity, r_half = (.5**(-2/3)-1)**-.5 * (10 | units.pc) )

integrator=bridge.Bridge(verbose=True,timestep=dt,use_threading=True)
integrator.add_system(gravity,(df_model,halo_model,), do_sync=True) # note the order we add kickers doesn't matter

# evolve the bridge to the requested time
time = 0 | units.myr
t_end = 5 | units.Gyr
count = 0

t = [] | units.Myr
r = [] | units.kpc
while time < t_end:
    time +=dt
    count+=1
    integrator.evolve_model(time)
    print('evolved to', time.in_(units.Myr)) 
    radius = gravity.particles.center_of_mass().length().in_(units.kpc)
    print('cluster distance from galactic centre', radius)
    t.append(time)
    r.append(radius)
gravity.stop()

plt.figure()
plot(t,r)
plt.savefig('trajectory.png')
