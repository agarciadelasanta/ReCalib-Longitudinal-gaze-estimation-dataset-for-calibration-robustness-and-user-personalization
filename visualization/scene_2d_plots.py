import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import cv2 as cv
import random 

def showFacePoints(Pts, imageSize):
    fig, ax = plt.subplots()
    # Screen frame
    scr = np.array([[0, 0], [imageSize[0], 0], [imageSize[0], imageSize[1]], [0, imageSize[1]], [0, 0]])
    ax.plot(scr[:,0], scr[:,1], color='black')
    ax.plot(scr[0,0], scr[0,1], color='red', marker = 'o')
    # Points
    ax.plot(Pts[:,0], Pts[:,1], color='blue', marker = '.')
    ax.scatter(Pts[0,0], Pts[0,1], color='red', marker = 'o')
    # Bonito
    ax.set_xlim([0, imageSize[0]])
    ax.set_ylim([0, imageSize[1]])
    ax.axis('equal')
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


class trackingScreen2DPlotter():
    def __init__(self, screenWpx, screenHpx, zoom, orientation=0) -> None:
        self.zoom = zoom
        self.screenWpx = int(screenWpx * self.zoom)
        self.screenHpx = int(screenHpx * self.zoom)

        self.desfase = (int(self.screenWpx/2), int(self.screenHpx/2) )

        self.circleV = []
        self.pointV = []
        self.pointX = []

        self.fig = np.zeros((self.screenHpx*2, self.screenWpx*2, 3), np.uint8)
        self.fig = cv.rectangle(self.fig, 
                        (int(self.screenWpx/2), int(self.screenHpx/2)), 
                        (int(self.screenWpx + (self.screenWpx/2)), int(self.screenHpx + (self.screenHpx/2))), 
                        (0, 255, 0), 3)
        
    def appendPoint(self, point, color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))):
        point = np.add(point, self.desfase)
        self.pointV.append((point, color))

    def appendCircle(self, point, radius, color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))):
        point = np.add(point, self.desfase)
        self.circleV.append((point, radius, color))

    def appendCamera(self, point, color = (random.randint(20, 150), random.randint(20, 150), random.randint(20, 150))):
        point = np.add(point*np.array([self.zoom,self.zoom]), self.desfase)
        self.pointX.append((point, color))


    def plot(self):
        for aux in self.circleV:
            addCircle(self.fig, aux[0], aux[1], aux[2])
        
        for aux in self.pointV:
            addPoint(self.fig, aux[0], aux[1])

        for aux in self.pointX:
            addCross(self.fig, aux[0], aux[1])

        cv.flip(self.fig, 1)
        return self.fig


def addCircle(fig, circle, radius, color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))):
    cv.ellipse(img=fig, center=circle.astype(int), axes=(radius, radius) , angle=0, startAngle=0, endAngle=360, color=color, thickness=8)
    return fig

def addPoint(fig, point, color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))):
    fig = cv.circle(img=fig, center=point.astype(int), radius=30, color=color, thickness=10)
    return fig

def addCross(fig, point, color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))):
    center = point.astype(int)
    size = 30
    thickness = 10
    # Draw vertical line
    fig = cv.line(fig, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness, lineType=cv.LINE_AA)
    # Draw horizontal line
    fig = cv.line(fig, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness, lineType=cv.LINE_AA)
    return fig

def show2D(fig):    
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=0.25, z=1.25)
    )    
    fig.update_layout(scene_camera=camera,
                      scene_aspectmode='data')
    plot(fig, auto_open=True)