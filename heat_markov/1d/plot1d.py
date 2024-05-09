import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    file_name = sys.argv[1]
    data = np.loadtxt(file_name)

    x = np.linspace(0, 1, data.shape[1])


    if data.ndim == 1:
        plt.plot(x, data)
        plt.show()
        plt.title('1D Heat Equation')
        plt.xlabel('Length of bar (m)')
        plt.ylabel('Temperature')
    else:
        fig, ax = plt.subplots()
        line, = ax.plot(x, data[0])

        ax.set_title('1D Heat Equation')
        ax.set_xlabel('Length of bar (m)')
        ax.set_ylabel('Temperature')

        def init():
            line.set_ydata(data[0, ::-1])
            return line,

        def animate(i):
            line.set_ydata(data[i, ::-1])
            return line,

        ani = FuncAnimation(fig, animate, frames=data.shape[0], init_func=init, interval=5)
        #ani.save('1d_heat.mp4', writer='ffmpeg')
        plt.show()
