"""
Real-Time Fluid Dynamics Simulation using Phiflow
=================================================
A side project developed to explore modern Python computational fluid dynamics (CFD) frameworks.
This script simulates smoke advection, buoyancy, and incompressibility (Navier-Stokes projection)
using the phi.flow library, coupled with a custom visual rendering pipeline in Matplotlib.

Author: Bastien Detemmerman
"""

import numpy as np
import matplotlib.pyplot as plt
from phi.flow import *

# 1. Définition de l'espace physique (Frontières ouvertes)
DOMAIN = dict(x=64, y=128, bounds=Box(x=(0, 100), y=(0, 200)))
smoke = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)
velocity = StaggeredGrid(0, extrapolation.BOUNDARY, **DOMAIN)
inflow_base = 0.5 * CenteredGrid(Sphere(x=50, y=10, radius=5), extrapolation.BOUNDARY, **DOMAIN)

# 2. Création du rayon de lumière (Le "Halo")
Y, X = np.mgrid[0:128, 0:64]
ray_center_x = 30 + (Y / 128.0) * 25
ray_width = 8.0
light_mask = np.exp(-((X - ray_center_x) ** 2) / (ray_width ** 2))
light_mask *= (Y / 128.0) * 0.9 + 0.1

# 3. Configuration de la fenêtre Matplotlib
plt.ion()
fig, ax = plt.subplots(figsize=(5, 8))
fig.patch.set_facecolor('black')

img = ax.imshow(light_mask * 0.15, cmap='gray', origin='lower', vmin=0, vmax=2.5)
ax.axis('off')

# 4. Boucle principale de simulation
for frame in range(300):
    # --- CALCULS PHYSIQUES ---

    # A. Injection de la matière (la fumée) via advection de MacCormack
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + inflow_base

    # B. Injection de l'air (LE SOUFFLE / L'IMPULSION)
    souffle = inflow_base * (0, 2.5) @ velocity
    velocity = velocity + souffle

    # C. Flottabilité et Advection de l'air (Semi-Lagrangien)
    buoyancy_force = smoke * (0, 0.3) @ velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force

    # D. Incompressibilité (Création des tourbillons / Projection Navier-Stokes)
    velocity, pressure = fluid.make_incompressible(velocity)

    # --- RENDU VISUEL (L'effet photo) ---
    smoke_data = smoke.values.numpy('y,x')
    illuminated_smoke = smoke_data * (light_mask * 4.0 + 0.1)
    final_render = illuminated_smoke + (light_mask * 0.15)

    img.set_data(final_render)
    plt.pause(0.01)

plt.ioff()
plt.show()
