# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:13:42 2021

@author: zeyna
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close("all")

# SV = np.random.normal(loc=0.6, scale=0.1, size=(200, 200))

# plate size, mm

w = h = 2

# intervals in x-, y- directions, mm

dx = dy = 0.01

# Thermal diffusivity of steel, mm2.s-1

D = 8.7 * 10 ** -5

Tcool, Thot = 0, 4

nx, ny = int(w / dx), int(h / dy)

dx2, dy2 = dx * dx, dy * dy
# why did we define dt like that
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

print("dt:", dt)

u0 = np.zeros((nx, ny))

u = u0.copy()

Cb0 = np.zeros((nx, ny))

Cb = Cb0.copy()

SV = np.ones((nx, ny))

SV[:, 0:int(ny / 2)] = 0.1

SV[:, int(ny / 2):ny] = 0.6


center_x = nx/2
center_y = ny/2
radius = 50
y,x = np.ogrid[-center_x:nx-center_x, -center_y:ny-center_y]
mask = x*x + y*y <= radius*radius

kon = np.full((nx, ny), 7.7 * 10 ** -1)
kon[mask] = 7.7

koff = np.full((nx, ny), 7.7 * 10 ** -4)
koff[mask] = 7.7 * 10 ** -3

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(kon, cmap=plt.get_cmap('hot'))
# ax2.imshow(koff, cmap=plt.get_cmap('hot'))
# plt.show()

#kon = 7.7 * 10 ** -1


#koff = 7.7 * 10 ** -4


# Initial conditions - circle of radius r centred at (cx,cy) (mm)

r, cx, cy = 0.5, 1, 1

r2 = r ** 2

Cv0 = 1
Lv = 3.3 * 10 ** -2
lamb = np.log(2) / 30
t0 = 0
x0 = 0
y0 = 0

R0 = np.random.normal(loc=4.089 * 10 ** -1, scale=8 * 10 ** -2, size=(nx, ny))


def do_timestep(u0, u, t0, Cb0, Cb):
    # Propagate with forward-difference in time, central-difference in space

    Cv = Cv0 * np.exp(-lamb * t0)

    u[1:-1, 1:-1] = (u0[1:-1, 1:-1]

                     + dt * D * ((u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2

                                 + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)

                     + dt * Lv * SV[1:-1, 1:-1] * (Cv) + koff[1:-1, 1:-1] * dt * Cb0[1:-1, 1:-1]) / (1. + dt * Lv * SV[1:-1, 1:-1]
                                                                                         + dt * kon[1:-1, 1:-1] * (R0[1:-1,
                                                                                                       1:-1] - Cb0[1:-1,
                                                                                                               1:-1]))

    Cb[1:-1, 1:-1] = Cb0[1:-1, 1:-1] - dt * koff[1:-1, 1:-1] * Cb0[1:-1, 1:-1] + dt * kon[1:-1, 1:-1] * u[1:-1, 1:-1] * (
            R0[1:-1, 1:-1] - Cb0[1:-1, 1:-1])

    t0 = t0 + dt
    u0 = u.copy()
    Cb0 = Cb.copy()

    return u0, u, t0, Cb0, Cb


def get_data(nsteps, ninterval, u0, u, t0, Cb0, Cb, visualize=False):
    # Number of timesteps

    nsteps = nsteps

    i_coords, j_coords = np.meshgrid(range(200), range(200), indexing='ij')

    coordinate_grid = np.array([i_coords, j_coords])

    input_array = []

    k = 0
    for m in range(nsteps):

        u0, u, t0, Cb, Cb0 = do_timestep(u0, u, t0, Cb0, Cb)

        if m % ninterval == 0:
            k += 1
            if visualize == True:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool, vmax=Thot)
                ax[1].imshow(Cb.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool, vmax=Thot)
                ax[0].set_title('{:.1f} ms'.format(m))
            plt.show()
            time_array = np.full((40000, 1), t0)
            single_time = np.column_stack((coordinate_grid[0].reshape(40000, 1), coordinate_grid[1].reshape(40000, 1),
                                           time_array, u.reshape(40000, 1), Cb.reshape(40000, 1)))
            input_array.append(single_time)

    num_items = int(40000 * (nsteps) / ninterval)

    np_input_array = np.array(input_array).reshape(num_items, 5)
    print("Ci min: ", np_input_array[:,3].min())
    print("Ci max: ", np_input_array[:,3].max())
    print("Cb min: ", np_input_array[:,4].min())
    print("Cb max: ", np_input_array[:,4].max())
    print(k)

    return np_input_array, num_items, kon, koff


def visualize_data(nsteps, u0, u, t0, Cb0, Cb):
    # Number of timesteps
    # Output 4 figures at these timesteps

    fignum = 0
    fig = plt.figure()
    for m in range(nsteps):
        u0, u, t0, Cb, Cb0 = do_timestep(u0, u, t0, Cb0, Cb)
        if m % 100 == 0:




            fig, ax = plt.subplots(1, 2)
            im0 = ax[0].imshow(u.copy(), cmap=plt.get_cmap('bwr'), vmin=0, vmax=0.3)
            ax[0].set_axis_off()

            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im0, cax=cax)

            im1 = ax[1].imshow(Cb.copy(), cmap=plt.get_cmap('ocean'), vmin=0, vmax=1)
            ax[1].set_axis_off()
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im1, cax=cax1)
            plt.show()

    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    # cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    # #fig.colorbar(im, cax=cbar_ax)
    # plt.show()





