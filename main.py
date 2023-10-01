import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import benchmarkfunctions as bf
import PSO

x = bf.x
y = bf.y
z = bf.matyas(x, y)
xL = bf.x_lower_bound
xU = bf.x_upper_bound
yL = bf.y_lower_bound
yU = bf.y_upper_bound
pbest = PSO.pbest
POS = PSO.POS
VELO = PSO.VELO
gbest = PSO.gbest
gbest_obj = PSO.gbest_obj

x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[xL, xU, yL, yU], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(POS[0], POS[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(POS[0], POS[1], VELO[0], VELO[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([xL,xU])
ax.set_ylim([yL,yU])

def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    title = 'Iteration {:02d}'.format(i)
    # Update params
    PSO.update()
    # Set picture
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(POS.T)
    p_arrow.set_offsets(POS.T)
    p_arrow.set_UVC(VELO[0], VELO[1])
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500)
anim.save("PSO.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
print("Global optimal at f({})={}".format([x_min,y_min], bf.matyas(x_min,y_min)))