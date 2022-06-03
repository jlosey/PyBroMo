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

sd = 101010 #int(sys.argv[1])
BG_A = 1800. #float(sys.argv[2])
BG_D = 1500. # float(sys.argv[3])
PSF_X = 0.3e-6 
PSF_Y = 0.3e-6
PSF_Z = 0.5e-6
def efficiency(r,r0):
    return [1./(1. + (r[i]/r0)**6) for i in range(len(r))]

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

n1 = 5 
n2 = 2
nn = (n1,n2) # list of populations, 2D needed 
dc50 = np.ones(n1)*50.
dc0 = np.ones(n2)*1000.
dc = np.hstack((dc50,dc0))
# Free Energy function and parameters
def bistable(x,x0,wid=15,spr=0.001):
    """Not specifically needed for simulation but here for reference"""
    diff = np.subtract(x,x0)
    return -(spr/4)(diff**2 - wid**2)**2
def der_bistable(x,x0,wid=15,spr=0.001):
    """derivative of bistable equation """
    diff = np.subtract(x,x0)
    return -spr*(diff)*((diff)**2-wid**2)
def f1(x):
    return der_bistable(x,x0=55,wid=10,spr=1e-4)
def f2(x):
    return der_bistable(x,x0=45,wid=10,spr=1e-4)
fe = (f1,f2)
dff = (D1,D1)
ldff = (2e-3,2e-3)

dye0 = None

# Particles definition
P = pbm.Particles.from_specs(num_particles=(n1,n2)
        ,D=(D1,D1)
        ,free_energy=fe
        ,D_L=ldff
        ,dye0=(None,None)
        ,box=box        
        ,rs=rs)
# Simulation time step (seconds)
t_step = 1e-7 #5e-8

# Time duration of the simulation (seconds)
t_max = 0.05

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
E = pbm.FRETEfficiency(S, populations = (n1+n2,),
                             em_rates = (200e3,),
                             E_method="theoretical",
                             R0=56.)
E.add_efficiency_max_rates(t_chunksize=2**19, chunkslice='times')
#Sfile = pbm.ParticlesSimulation.from_datafile(hsh, mode='w')
params1 = dict(
    em_rates = (200e3,), #em_list,    # Peak emission rates (cps) for each population (D+A)
    #E_values = E,     # FRET efficiency for each population
    num_particles = (n1+n2,), #p_list,   # Number of particles in each population
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
#S.store.close()
#E.fret_store.close()
#sim1.save_photon_hdf5()
np.savetxt(f"smalltest_bistable50_theoretical_{sd}_{BG_A}_{BG_D}.txt"
        ,np.column_stack((ts_1,ts_1*t_step,dt_1,p_1))
        ,fmt=['%d','%f','%d','%d']
        ,header = str1.lstrip()
)
