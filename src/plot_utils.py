import numpy as np
import pandas as pd
from collections import Counter
import warnings
from pandas.errors import SettingWithCopyWarning
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.graph_objs as go
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def pitch():
    """
    code to plot a soccer pitch 
    """
    # Create figure
    fig,ax=plt.subplots(figsize=(7,5))
    
    # Pitch Outline & Centre Line
    plt.plot([0,0],[0,100], color="black")
    plt.plot([0,100],[100,100], color="black")
    plt.plot([100,100],[100,0], color="black")
    plt.plot([100,0],[0,0], color="black")
    plt.plot([50,50],[0,100], color="black")

    # Left Penalty Area
    plt.plot([16.5,16.5],[80,20],color="black")
    plt.plot([0,16.5],[80,80],color="black")
    plt.plot([16.5,0],[20,20],color="black")

    # Right Penalty Area
    plt.plot([83.5,100],[80,80],color="black")
    plt.plot([83.5,83.5],[80,20],color="black")
    plt.plot([83.5,100],[20,20],color="black")

    # Left 6-yard Box
    plt.plot([0,5.5],[65,65],color="black")
    plt.plot([5.5,5.5],[65,35],color="black")
    plt.plot([5.5,0.5],[35,35],color="black")

    # Right 6-yard Box
    plt.plot([100,94.5],[65,65],color="black")
    plt.plot([94.5,94.5],[65,35],color="black")
    plt.plot([94.5,100],[35,35],color="black")

    # Prepare Circles
    centreCircle = Ellipse((50, 50), width=30, height=39, edgecolor="black", facecolor="None", lw=1.8)
    centreSpot = Ellipse((50, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    leftPenSpot = Ellipse((11, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    rightPenSpot = Ellipse((89, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    # Limit axes
    plt.xlim(0,100)
    plt.ylim(0,100)
    
    ax.annotate("", xy=(25, 5), xytext=(5, 5),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.text(7,7,'Attack',fontsize=20)
    return fig,ax

def draw_pitch(pitch, line, orientation='h', view='full', alpha=1, size_x=10.4, size_y=6.8):
    """
    Draw a soccer pitch given the pitch, the orientation, the view and the line
    
    Parameters
    ----------
    pitch
    
    """
    orientation = orientation
    view = view
    line = line
    pitch = pitch
    
    if orientation.lower().startswith("h"):
        
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(size_x, size_y * 2))
            plt.xlim(51,105)
            plt.ylim(-1,69)
        else:
            fig,ax = plt.subplots(figsize=(size_x, size_y))
            plt.xlim(-1,105)
            plt.ylim(-1,69)
        ax.axis('off')  # this hides the x and y ticks
    
        # side and goal lines
        ly1 = [0,0,68,68,0]
        lx1 = [0,104,104,0,0]

        plt.plot(lx1,ly1,color=line,zorder=1)

        # boxes, 6 yard box and goals

        # outer boxes
        ly2 = [13.84,13.84,54.16,54.16] 
        lx2 = [104,87.5,87.5,104]
        plt.plot(lx2,ly2,color=line,zorder=1)

        ly3 = [13.84,13.84,54.16,54.16] 
        lx3 = [0,16.5,16.5,0]
        plt.plot(lx3,ly3,color=line,zorder=1)

        # goals
        ly4 = [30.34,30.34,37.66,37.66]
        lx4 = [104,104.2,104.2,104]
        plt.plot(lx4,ly4,color=line,zorder=1)

        ly5 = [30.34,30.34,37.66,37.66]
        lx5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=1)


        # 6 yard boxes
        ly6 = [24.84,24.84,43.16,43.16]
        lx6 = [104,99.5,99.5,104]
        plt.plot(lx6,ly6,color=line,zorder=1)

        ly7 = [24.84,24.84,43.16,43.16]
        lx7 = [0,4.5,4.5,0]
        plt.plot(lx7,ly7,color=line,zorder=1)

        # halfway line, penalty spots, and kickoff spot
        ly8 = [0,68] 
        lx8 = [52,52]
        plt.plot(lx8,ly8,color=line,zorder=1)


        plt.scatter(93,34,color=line,zorder=1)
        plt.scatter(11,34,color=line,zorder=1)
        plt.scatter(52,34,color=line,zorder=1)

        circle1 = plt.Circle((93.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=0,alpha=1)
        circle2 = plt.Circle((10.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=0,alpha=1)
        circle3 = plt.Circle((52, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)

        # rectangles in boxes
        rec1 = plt.Rectangle((87.5,20), 16,30,ls='-',color=pitch, zorder=0,alpha=alpha)
        rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=0,alpha=alpha)

        # pitch rectangle
        rec3 = plt.Rectangle((-1, -1), 106,70,ls='-',color=pitch, zorder=0,alpha=alpha)

        ax.add_artist(rec3)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle3)
        
    else:
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(size_y * 2, size_x))
            plt.ylim(51,105)
            plt.xlim(-1,69)
        else:
            fig,ax = plt.subplots(figsize=(size_y, size_x))
            plt.ylim(-1,105)
            plt.xlim(-1,69)
        ax.axis('off')  # this hides the x and y ticks

        # side and goal lines
        lx1 = [0,0,68,68,0]
        ly1 = [0,104,104,0,0]

        plt.plot(lx1,ly1,color=line,zorder=1)

        # boxes, 6 yard box and goals

        # outer boxes
        lx2 = [13.84,13.84,54.16,54.16] 
        ly2 = [104,87.5,87.5,104]
        plt.plot(lx2,ly2,color=line,zorder=1)

        lx3 = [13.84,13.84,54.16,54.16] 
        ly3 = [0,16.5,16.5,0]
        plt.plot(lx3,ly3,color=line,zorder=1)

        # goals
        lx4 = [30.34,30.34,37.66,37.66]
        ly4 = [104,104.2,104.2,104]
        plt.plot(lx4,ly4,color=line,zorder=1)

        lx5 = [30.34,30.34,37.66,37.66]
        ly5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=1)


        # 6 yard boxes
        lx6 = [24.84,24.84,43.16,43.16]
        ly6 = [104,99.5,99.5,104]
        plt.plot(lx6,ly6,color=line,zorder=1)

        lx7 = [24.84,24.84,43.16,43.16]
        ly7 = [0,4.5,4.5,0]
        plt.plot(lx7,ly7,color=line,zorder=1)

        # halfway line, penalty spots, and kickoff spot
        lx8 = [0,68] 
        ly8 = [52,52]
        plt.plot(lx8,ly8,color=line,zorder=1)


        plt.scatter(34,93,color=line,zorder=1)
        plt.scatter(34,11,color=line,zorder=1)
        plt.scatter(34,52,color=line,zorder=1)

        circle1 = plt.Circle((34,93.5), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=0,alpha=1)
        circle2 = plt.Circle((34,10.5), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=0,alpha=1)
        circle3 = plt.Circle((34,52), 9.15, ls='solid', lw=1.5, color=line, fill=False, zorder=1,alpha=1)


        # rectangles in boxes
        rec1 = plt.Rectangle((20, 87.5), 30,16.5,ls='-',color=pitch, zorder=0,alpha=alpha)
        rec2 = plt.Rectangle((20, 0), 30,16.5,ls='-',color=pitch, zorder=0,alpha=alpha)

        # pitch rectangle
        rec3 = plt.Rectangle((-1, -1), 70,106,ls='-',color=pitch, zorder=0,alpha=alpha)

        ax.add_artist(rec3)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle3)

    return fig, ax


def get_pitch_layout(title):
    xmax = 104
    ymax = 68
    lines_color = 'black'
    bg_color = 'rgb(255, 255, 255)'

    pitch_layout = dict(
        hovermode='closest', autosize=False,
        width=825,
        height=600,
        plot_bgcolor=bg_color,  # 'rgb(59, 205, 55)',
        xaxis={
            'range': [0, xmax],
            'showgrid': False,
            'showticklabels': False,
        },
        yaxis={
            'range': [0, ymax],
            'showgrid': False,
            'showticklabels': False,
        },
        title=title,
        shapes=[
            # Center Circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.35,
                'x0': xmax * 0.40,
                'y1': ymax * 0.65,
                'x1': xmax * 0.60,
                'line': {'color': lines_color,},
            },
            # Left Penalty Area
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.165,
                'x0': 0,
                'y1': ymax * 0.165,
                'x1': xmax * 0.200,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.165,
                'x0': xmax * 0.200,
                'y1': ymax * 0.835,
                'x1': xmax * 0.200,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.835,
                'x0': xmax * 0.200,
                'y1': ymax * 0.835,
                'x1': 0,
                'line': {'color': lines_color,},
            },
            # Right Penalty Area
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.165,
                'x0': xmax,
                'y1': ymax * 0.165,
                'x1': xmax * 0.800,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.165,
                'x0': xmax * 0.800,
                'y1': ymax * 0.835,
                'x1': xmax * 0.800,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.835,
                'x0': xmax * 0.800,
                'y1': ymax * 0.835,
                'x1': xmax,
                'line': {'color': lines_color,},
            },
            # Left 6-yard Box
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.35,
                'x0': 0,
                'y1': ymax * 0.35,
                'x1': xmax * 0.055,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.35,
                'x0': xmax * 0.055,
                'y1': ymax * 0.65,
                'x1': xmax * 0.055,
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.65,
                'x0': xmax * 0.055,
                'y1': ymax * 0.65,
                'x1': 0,
                'line': {'color': lines_color,},
            },
            # Right 6-yard Box
            {   
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.35,
                'x0': xmax,
                'y1': ymax * 0.35,
                'x1': xmax * 0.945,                                         
                'line': {'color': lines_color,},
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.35,
                'x0': xmax * 0.945,
                'y1': ymax * 0.65,
                'x1': xmax * 0.945,
                'line': {
                    'color': lines_color,
                }
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax * 0.65,
                'x0': xmax * 0.945,
                'y1': ymax * 0.65,
                'x1': xmax,
                'line': {
                    'color': lines_color,
                }
            },
            # Pitch Outline & Center Line
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax,
                'x0': xmax * 0.5,
                'y1': 0,
                'x1': xmax * 0.5,
                'line': {
                    'color': lines_color,
                }
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': 0,
                'x0': 0,
                'y1': ymax,
                'x1': 0,
                'line': {
                    'color': lines_color,
                }
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': 0,
                'x0': xmax,
                'y1': ymax,
                'x1': xmax,
                'line': {
                    'color': lines_color,
                }
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': ymax,
                'x0': 0,
                'y1': ymax,
                'x1': xmax,
                'line': {
                    'color': lines_color,
                }
            },
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'y',
                'y0': 0,
                'x0': 0,
                'y1': 0,
                'x1': xmax,
                'line': {
                    'color': lines_color,
                }
            },
        ]
    )
    return pitch_layout  