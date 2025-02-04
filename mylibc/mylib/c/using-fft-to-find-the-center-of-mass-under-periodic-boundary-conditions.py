import numpy as np
import matplotlib.pyplot as plt

Points_x  = np.random.randn(10000)+1
plt.plot(Points_x)
plt.title("Start")
plt.show()

Box_min   = -10
Box_max   =  10
Box_width = Box_max - Box_min

#Maps Points to Box_min ... Box_max with periodic boundaries
Points_x = (Points_x%Box_width + Box_min)
plt.plot(Points_x)
plt.title("Periodic boundaries")
plt.show()

#Map Points to -pi..pi
Points_map = (Points_x - Box_min)/Box_width*2*np.pi-np.pi
plt.plot(Points_map)
plt.title("-pi...pi")
plt.show()

#Calc circular mean
Pmean_map  = np.arctan2(np.sin(Points_map).mean() , np.cos(Points_map).mean())
#Map back
Pmean = (Pmean_map+np.pi)/(2*np.pi) * Box_width + Box_min

#Plotting the result
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.hist(Points_x, 100);
plt.plot([Pmean, Pmean], [0, 1000], c='r', lw=3, alpha=0.5);
plt.subplot(122,aspect='equal')
plt.plot(np.cos(Points_map), np.sin(Points_map), '.');
plt.ylim([-1, 1])
plt.xlim([-1, 1])
plt.grid()
plt.plot([0, np.cos(Pmean_map)], [0, np.sin(Pmean_map)], c='r', lw=3, alpha=0.5);
plt.show()
