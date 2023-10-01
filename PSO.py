import numpy as np
import benchmarkfunctions as bf

# Hyper-parameter of the algorithm
c1 = c2 = 1.2
#1.2 0.2 2.2
w = 0.75
#0.75 0.25 1.25
# Create particles
n_particles = 30
#30 15 45
np.random.seed(100)
POS = np.random.rand(2, n_particles) * 10
VELO = np.random.randn(2, n_particles) * 0.1

# Initialize data
pbest = POS
pbest_obj = bf.matyas(POS[0], POS[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

def update():
    "Function to do one iteration of particle swarm optimization"
    global VELO, POS, pbest, pbest_obj, gbest, gbest_obj
    # Update params
    r1, r2 = np.random.rand(2)
    VELO = w * VELO + c1*r1*(pbest - POS) + c2*r2*(gbest.reshape(-1,1)-POS)
    POS = POS + VELO
    obj = bf.matyas(POS[0], POS[1])
    pbest[:, (pbest_obj >= obj)] = POS[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
