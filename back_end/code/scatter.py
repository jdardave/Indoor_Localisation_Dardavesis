from re import X
from tkinter.ttk import Notebook
from mpl_toolkits import mplot3d
# %matplotlib Notebook
import numpy as np
import matplotlib.pyplot as plt

f = open("../back_end/data/808_colored.txt", 'r')

fig = plt.figure()
figsize=(3,4)
ax = plt.axes(projection='3d')

x=[]
y=[]
z=[]
x1=[]
y1=[]
z1=[]

for coords in f:
    coords=coords.strip('\n')
    coords=coords.split(' ')
    x.append(coords[0])
    y.append(coords[1])
    z.append(coords[2])
    print(x,y,z)    
# x.remove('')

# print(y)
for element in x:
    x1.append(float(element))

for element in y:
    y1.append(float(element))

for element in z:
    z1.append(float(element))

# print(x1[0],y1[0],z1[0])
# ax.set_xlim([x1[0]-2,x1[0]+2])
# ax.set_ylim([y1[0]-2,y1[0]+2])
# ax.set_zlim([z1[0]-2,z1[0]+2])
ax.scatter3D(x1[0], z1[0], y1[0], c='b',s=20,label="Reference")
ax.scatter3D(x1[1:], z1[1:], y1[1:], c='r',s=20,label="User",alpha=1)
ax.set_title('Scatter plot of point cloud centers')
ax.legend(loc=4,fontsize=16)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
plt.show()

