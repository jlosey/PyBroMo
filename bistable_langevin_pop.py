import sys
from pathlib import Path
from textwrap import dedent, indent
import numpy as np
import tables
import matplotlib.pyplot as plt
#import seaborn as sns
import pybromo as pbm
from scipy.stats import expon
import phconvert as phc
print('PyTables version:', tables.__version__)
print('PyBroMo version:', pbm.__version__)

sd = int(sys.argv[1])
POP = int(sys.argv[2])
POP_DO = int(sys.argv[3])
BG_A = float(sys.argv[4])
BG_D = float(sys.argv[5])
PSF_X = float(sys.argv[6]) #0.3e-6 
PSF_Y = float(sys.argv[7]) #0.3e-6
PSF_Z = float(sys.argv[8]) #0.5e-6

#def efficiency(r,r0):
#    return [1./(1. + (r[i]/r0)**6) for i in range(len(r))]

# Initialize the random state
rs = np.random.RandomState(seed=sd)
print('Initial random state:', pbm.hashfunc(rs.get_state()))

# kbT
kb = 1.380649e-23
T = 300
kbT = kb*T
beta = 1./kbT
D1 = 30.*(1e-6)**2

# Diffusion coefficient
#Du = 12.0            # um^2 / s
#D1 = Du*(1e-6)**2    # m^2 / s
D2 = D1

# Simulation box definition
#box = pbm.Box(x1=-4.e-7, x2=4.e-7, y1=-4.e-7, y2=4.e-7, z1=-6e-7, z2=6e-7)
box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)
#box = pbm.Box(x1=-2.e-6, x2=2.e-6, y1=-2.e-6, y2=2.e-6, z1=-3e-6, z2=3e-6)
#box = pbm.Box(x1=-1.e-6, x2=1.e-6, y1=-1.e-6, y2=1.e-6, z1=-2e-6, z2=2e-6)

# PSF definition
psf = pbm.GaussianPSF(sx=PSF_X,sy=PSF_Y,sz=PSF_Z)
#psf = pbm.NumericPSF()

# Particles definition
n1 = POP 
n2 = POP_DO
nn = n1+n2
dc50 = np.ones(n1)*50.
dc0 = np.ones(n2)*1000.
dc = np.hstack((dc50,dc0))
P = pbm.Particles.from_specs(num_particles=(nn,)
        ,D=(D1,)
        ,dye_center=(dc,)
        ,box=box        
        ,rs=rs)

# Simulation time step (seconds)
t_step = 5e-8

# Time duration of the simulation (seconds)
t_max = 30. 

# Langevin Parameters for bistable
k_bs = 1e-4
rc_bs = 50.
w_bs = 7.5
diff_langevin = 2e-3 
bistable = lambda r: (k_bs/4)*((r-r_bs)**2 - (w_bs)**2)**2

# Particle simulation definition
S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, T=T, 
                            particles=P, box=box, psf=psf,
                            )

print('Current random state:', pbm.hashfunc(rs.get_state()))
S.simulate_diffusion(total_emission=False, save_pos=False, verbose=True,
                     rs=rs, chunksize=2**19, chunkslice='times')
print('Current random state:', pbm.hashfunc(rs.get_state()))
print(S.compact_name())
#hsh = S.hash()[:6]
#S.store.close()
E = pbm.FRETEfficiency(S, populations = (nn,),
                             em_rates = (200e3,),
                             E_method="theoretical",
                             R0=56.,
                             )
E.add_efficiency_max_rates(t_chunksize=2**19, chunkslice='times')
#Sfile = pbm.ParticlesSimulation.from_datafile(hsh, mode='w')
params1 = dict(
    em_rates = (200e3,), #em_list,    # Peak emission rates (cps) for each population (D+A)
    #E_values = E,     # FRET efficiency for each population
    num_particles = (nn,), #p_list,   # Number of particles in each population
    bg_rate_d = BG_D, #1800,       # Poisson background rate (cps) Donor channel
    bg_rate_a = BG_A, #1200,        # Poisson background rate (cps) Acceptor channel
    #E_method = "theoretical",
    #R0 = 56.
    )
sim1 = pbm.TimestampSimulation(S, E, **params1)
#sim1.summarize()
#rs = np.random.RandomState(sd)
sim1.run(rs=rs, 
            overwrite=True,      # overwite existing timstamp arrays
            skip_existing=True,  # skip simulation of existing timestamps arrays to save time
            save_pos=False,      # save particle position at emission time
	    chunksize=2**19,
           )
sim1.merge_da()
str1 = sim1.__str__()+"\nTimestep,Time,Channel(0=D,1=A),ParticleNum"
ts_1 = sim1.ts 
dt_1 = sim1.a_ch.astype('uint8')
p_1 = sim1.part
S.ts_store.close()
S.store.close()
E.fret_store.close()
#sim1.save_photon_hdf5()
np.savetxt(f"timestamp_bs_seed-{sd}_pop-{n1}_popDO-{n2}_bgA-{BG_A}_bgD{BG_D}_psf_x{PSF_X}_y{PSF_Y}_z{PSF_Z}.txt"
        ,np.column_stack((ts_1,ts_1*t_step,dt_1,p_1))
        ,fmt=['%d','%f','%d','%d']
        ,header = str1.lstrip()
)
