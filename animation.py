import numpy as np
from matplotlib import animation, pyplot as plt


def create_animation(title, state, speed=50, cmap='plasma'):
    frame_num = int(len(state) / speed)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    ax.set_title(title)
    img = ax.imshow(state[0], cmap=cmap)
    img.set_clim(np.max(state), np.min(state))
    plt.colorbar(mappable=img, ax=ax)

    def animate(i):
        data = state[i * speed]
        img.set_data(data)

    anim = animation.FuncAnimation(fig, animate, frames=frame_num)
    anim.save('{}.mp4'.format(title))
