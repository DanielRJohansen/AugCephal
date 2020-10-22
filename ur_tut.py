from ursina import *
import numpy as np
#cube = Entity(model="cube", color=color.orange, scale=(2, 2, 2))
cubes = []
from time import time as t

randobject = abs(np.random.rand(50,512,512))*255

print(randobject)



def update():
    for entity in cubes:                             # Go through the cube list
        camera.position += (0, time.dt*0.01,0)
def go():

    app = Ursina()

    window.title = 'My Game'  # The window title
    window.borderless = False  # Show a border
    window.fullscreen = False  # Do not go Fullscreen
    window.exit_button.visible = False  # Do not show the in-game red X that loses the window
    window.fps_counter.enabled = True

    t0 = t()
    for z in range(50):
        print(z)
        for y in range(512):
            for x in range(512):
                alpha = randobject[z,y,x]
                cubes.append(Entity(model='cube', color=color.rgba(250, 30, 30, alpha), position=(x-2,y-2,z-2), scale=(0.9,0.9,0.9)))
    print("Done: ", t()-t0)
    app.run()