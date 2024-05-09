import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    # get filename from command line
    filename = sys.argv[1]

    # load data
    data = np.loadtxt(filename)
    num_frames = data.shape[0] // data.shape[1]
    data = data.reshape((num_frames, data.shape[1], data.shape[1]))

    # animate the results of the heat transfer and show it as a movie, so one can see how the
    # temperature changes in the plate over time.

    
    fig = plt.figure()
    im = plt.imshow(data[0,:,:].T, cmap ='bwr', vmin = -1, vmax = 1, extent= [0,1,0,1])
    plt.xlabel('Horizontal plate edge (m)', fontsize = 10)
    plt.ylabel('Vertical plate edge (m)', fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.title('Temperature Map')
    plt.colorbar()

    def init():
        im.set_array(data[0,:,:].T)
        return im

    #Define the animation update function.  In this function, each map image will be updated with the current frame.
    def animate(i):
        im.set_array(data[i,:,:].T)
        return im
        
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = data.shape[0], interval = 5)
    #anim.save('heatmap_animation.mp4')

    # plot data
    plt.show()
