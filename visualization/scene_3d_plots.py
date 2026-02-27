import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
#from plotly.subplots import make_subplots

class trackingEnviroment3DPlotter():
    def __init__(self) -> None:
        self.axisV = []
        self.gazeV = []
        self.infGazeV = []
        self.screenV = []
        self.headV = []
        self.pointV = []
        self.circleV = []
        self.fig = go.Figure()
        self.lineLength = 150

    def appendAxis(self, axisAux, name):
        self.axisV.append((axisAux,name))

    def appendHead(self, headAux, name):
        self.headV.append((headAux, name))

    def appendScreen(self, screenAux, name):
        self.screenV.append((screenAux,name))

    def appendGaze(self, point1, point2, name):
        self.gazeV.append((point1, point2, name))

    def appendInfGaze(self, point, vector, name):
        self.infGazeV.append((point, vector, name))

    def appendPoint(self, point, name):
        self.pointV.append((point.reshape(1, 3), name))

    def appendCircle(self, point, radius, name):
        self.circleV.append((np.squeeze(point), radius, name))

    def plot(self):
        for aux in self.axisV:
            addAxis(self.fig, aux[0], aux[1])

        for aux in self.screenV:
            addRectangle(self.fig, aux[0], aux[1])

        for aux in self.gazeV:
            addLine(self.fig, np.vstack((aux[0], aux[1])), aux[2])

        for aux in self.headV:
            self.lineLength = aux[0][:,2]*1.5
            addHead(self.fig, aux[0], aux[1])

        for aux in self.infGazeV:
            addInfGaze(self.fig, aux[0], aux[1], aux[2], self.lineLength)

        for aux in self.pointV:
            addPoint(self.fig, aux[0], aux[1])

        for aux in self.circleV:
            addCircle(self.fig, aux[0], aux[1], aux[2])

        show3D(self.fig)
    
def init3D():
    fig = go.Figure()
#    fig = make_subplots(rows=1, cols=2)
    return fig

def addInfGaze(fig, point, vector, title, lineLength):
    # Generate points along the line
    t = np.linspace(-10, lineLength, 150)  # Range of t-values
    line_points = point + np.outer(t, vector)  # Compute line points

    # Create a trace for the line
    line_trace = go.Scatter3d(
        x=line_points[:, 0],
        y=line_points[:, 1],
        z=line_points[:, 2],
        mode='lines',
        line=dict(color='orange', width=2),
        name=title
    )
    fig.add_trace(line_trace)

def addCircle(fig, point, radius, title):
    # Create a circle in 3D space
    theta = np.linspace(0, 2 * np.pi, 100)
    center = np.array(point)  # Center coordinates

    # Calculate circle points with the specified center
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] * np.ones_like(theta)

    line_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name=title,
        mode='lines',      # Use 'lines' mode to connect the points
        line=dict(
            color='blue',   # Line color
            width=3          # Line width
        )
    )

    fig.add_trace(line_trace)


def addAxis(fig, axisMat, title):
    line_width = 10
    
    if 'pred' in title:
        line_width = 5

    fig.add_trace(go.Scatter3d(x=np.array(axisMat[0,0]), y=np.array(axisMat[0,1]), z=np.array(axisMat[0,2]),
                        mode='markers',
                        name=title,
                        marker_color='rgb(0, 0, 0)'))
    fig.add_trace(go.Scatter3d(x=axisMat[[0,1],0], y=axisMat[[0,1],1], z=axisMat[[0,1],2],
                        mode='lines',
                        name='X',
                        line=dict(width=line_width),
                        marker_color='rgb(255, 0, 0)'))    
    fig.add_trace(go.Scatter3d(x=axisMat[[0,2],0], y=axisMat[[0,2],1], z=axisMat[[0,2],2],
                        mode='lines',
                        name='Y',
                        line=dict(width=line_width),
                        marker_color='rgb(0, 255, 0)'))    
    fig.add_trace(go.Scatter3d(x=axisMat[[0,3],0], y=axisMat[[0,3],1], z=axisMat[[0,3],2],
                        mode='lines',
                        name='Z',
                        line=dict(width=line_width),
                        marker_color='rgb(0, 0, 255)'))

def addPoint(fig, point, title):
    point_color = 'rgb(10, 10, 10)'

    if 'pred' in title:
        point_color = 'rgb(5, 5, 5)'
    fig.add_trace(go.Scatter3d(x=np.array(point[0,0]), y=np.array(point[0,1]), z=np.array(point[0,2]),
                    mode='markers',
                    name=title,
                    marker_color=point_color))

def addRectangle(fig, rectan, title):
    rectan = np.vstack((rectan, rectan[0,:]))
    fig.add_trace(go.Scatter3d(x=np.array(rectan[:,0]), y=np.array(rectan[:,1]), z=np.array(rectan[:,2]),
                        mode='lines',
                        name=title,
                        line=dict(width=4),
                        marker_color='rgb(0, 0, 0)'))


def addHead(fig, pts, title):
    point_color = 'rgb(0, 0, 255)'
    point_width = 4
        
    if 'pred' in title:
        point_color = 'rgb(255, 0, 0)'
        point_width = 2

    fig.add_trace(go.Scatter3d(x=np.array(pts[:,0]), y=np.array(pts[:,1]), z=np.array(pts[:,2]),
                        mode='lines+markers',
                        name=title,
                        line=dict(width=1),
                        marker=dict(size=point_width),
                        marker_color=point_color))
    fig.add_trace(go.Scatter3d(x=np.array(pts[0,0]), y=np.array(pts[0,1]), z=np.array(pts[0,2]),
                        mode='markers',
                        name=title,
                        line=dict(width=point_width+1),
                        marker=dict(size=point_width),
                        marker_color=point_color))
    


def addLine(fig, line, title):
    line_width = 4
    line_color = 'rgb(0, 0, 255)'

    if 'pred' in title:
        line_color = 'rgb(255, 0, 255)'
        line_width = 2

    fig.add_trace(go.Scatter3d(x=line[:,0], y=line[:,1], z=line[:,2],
                        mode='lines',
                        name=title,
                        line=dict(width=line_width),
                        marker=dict(size=4),
                        marker_color=line_color))    



def show3D(fig):    
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=0.25, z=1.25)
    )    
    fig.update_layout(scene_camera=camera,
                      scene_aspectmode='data')
    plot(fig, auto_open=True)


#fig.update_layout(scene = dict(
#        xaxis = dict(nticks=4, range=[-100,100],),
#                     yaxis = dict(nticks=4, range=[-50,100],),
#                     zaxis = dict(nticks=4, range=[-100,100],),),
#                     width=700,
#                     margin=dict(r=20, l=10, b=10, t=10))
#                      yaxis=dict(scaleanchor="x", scaleratio=1),


