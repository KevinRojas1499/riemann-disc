import os
import pandas
import time
import torch
import math
import numpy as np

from torch import nn, Tensor
import plotly.graph_objects as go
import pandas as pd

# flow_matching
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver, RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Sphere, Manifold
from models import get_model
# Discrete
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm



if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(42)

# Plotting 

def plot_interactive_globe(points, classes=None, class_names=None):
    fig = go.Figure()
    
    lat = points[:,0]
    lon = points[:,1]
    
    if classes is not None:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta']
        marker_color = classes
        colorscale = None
        
        unique_classes = sorted(np.unique(classes))
        if len(unique_classes) <= len(colors):
            colorscale = [[i/len(unique_classes), colors[i % len(colors)]] 
                            for i in range(len(unique_classes)+1)]
            
            if class_names is not None:
                fig = go.Figure()
                
                for i, class_id in enumerate(unique_classes):
                    class_id = int(class_id)
                    mask = classes == class_id
                    name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    
                    # Ensure we have points for this class
                    if np.any(mask):
                        fig.add_trace(go.Scattergeo(
                            lon=lon[mask],
                            lat=lat[mask],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=colors[i % len(colors)],
                                symbol='circle'
                            ),
                            name=name,
                            showlegend=True
                        ))
                
                # Skip the main trace addition below since we've added individual traces
                marker_color = None
    else:
        marker_color = 'red'
        colorscale = None
    
    # Add the main trace if we haven't added individual traces for the legend
    if marker_color is not None:
        fig.add_trace(go.Scattergeo(
            lon=lon,
            lat=lat,
            mode='markers',
            marker=dict(
                size=5, 
                color=marker_color,
                colorscale=colorscale,
                symbol='circle',
                showscale=classes is not None and class_names is None
            )
        ))
    
    fig.update_geos(
        projection_type="orthographic",  # Creates a globe effect
        showland=True,                   # Show landmasses
        landcolor="rgb(217, 217, 217)",  # Gray land color
        showocean=True,
        oceancolor="rgb(0, 102, 204)",   # Ocean color
        showlakes=True,
        lakecolor="rgb(0, 102, 204)",
        showcountries=True,              # Show country borders
        countrycolor="black"
    )
    
    fig.update_layout(
        title="Interactive 3D Globe with Data Points",
        geo=dict(
            showframe=False,             # Remove outer box
            showcoastlines=True,
            coastlinecolor="black"
        ),
        # Explicitly set legend properties to ensure visibility
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant"
        ),
        showlegend=True  # Explicitly enable legend
    )
    
    fig.show()


class EarthDataset(torch.utils.data.Dataset):
    """Some Information about EarthDataset"""
    def __init__(self):
        super(EarthDataset, self).__init__()
        path_dir = '../data/'
        files = ['volerup.csv', 'fire.csv', 'flood.csv', 'quakes_all.csv']
        self.class_names = ['Eruption', 'Fire', 'Flood', 'Earthquake']
        skip_header = [2, 1, 2, 4]
        data_arr = []
        conds_arr = []
        for i, (f, head) in enumerate(zip(files, skip_header)):
            data = np.genfromtxt(os.path.join(path_dir, f), delimiter=",", skip_header=head)
            data_arr.append(data)
            conds_arr.append(np.ones(data.shape[0]) * i)
        
        self.data = self.lat_long_to_cartesian(np.concatenate(data_arr, axis=0))
        self.conds = np.concatenate(conds_arr, axis=0)

    def __getitem__(self, index):
        return self.data[index], self.conds[index]

    def __len__(self):
        return len(self.data)


    def lat_long_to_cartesian(self, raw_data):
        data_normalized= raw_data/180 * np.pi
        colat = np.pi / 2 - data_normalized[:,0]
        long = data_normalized[:,1] + np.pi
        return np.stack(
            [
                np.sin(colat) * np.cos(long),
                np.sin(colat) * np.sin(long),
                np.cos(colat),
            ],
            axis=-1,
        )

    def cartesian_to_lat_long(self, points):
        long = np.arctan2(points[:, 1], points[:, 0])
        long = np.where(long < 0, long + 2 * np.pi, long)
        colat = np.arccos(points[:, 2])
        lon = long - np.pi
        lat = np.pi/2 - colat
        converted =  np.stack([lat, lon], axis=-1)
        return converted / np.pi * 180

def wrap(manifold, samples):
    center = torch.cat([torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1)
    samples = torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2

    return manifold.expmap(center, samples)

if __name__ == '__main__':
    ds = EarthDataset()
    lr = 0.001
    batch_size = 256
    iterations = 20001
    print_every = 1000
    manifold = Sphere()
    dim = 3
    hidden_dim = 512
    vocab_size = len(ds.class_names) + 1
    print('vocab ', vocab_size)

    # velocity field model init
    vf = get_model('mlp', manifold, dim, hidden_dim, depth=6, vocab_size=vocab_size, context_len=1).to(device)

    # instantiate an affine path object
    path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

    # init optimizer
    optim = torch.optim.Adam(vf.parameters(), lr=lr)

    # train
    start_time = time.time()
    k = 0
    keep_going = True
    scheduler = PolynomialConvexScheduler(n=2.0)
    disc_path = MixtureDiscreteProbPath(scheduler=scheduler)
    disc_loss_fn = MixturePathGeneralizedKL(path=disc_path)
    mask_token = vocab_size - 1
    while keep_going:
        dl = dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True)
        for (data,y_1) in dl:
            k+=1
            if k > iterations:
                keep_going = False
                break
            optim.zero_grad() 

            y_1 = y_1.to(device).long().view(-1,1)
            y_0 = torch.zeros_like(y_1) + mask_token

            x_1 = data.to(device).float()
            x_0 = torch.randn_like(x_1[:,:-1]).to(device)
            x_0 = wrap(manifold, x_0)

            t = torch.rand(x_1.shape[0]).to(device) 

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            d_path_sample = disc_path.sample(t=t, x_0=y_0, x_1=y_1)

            drift, logits = vf(x=path_sample.x_t,y=d_path_sample.x_t, t=path_sample.t, s=d_path_sample.t)
            loss_r = torch.pow( drift - path_sample.dx_t, 2).mean()
            loss_d = disc_loss_fn(logits=logits, x_1=y_1, x_t=d_path_sample.x_t, t=d_path_sample.t)

            loss = loss_r + loss_d

            # optimizer step
            loss.backward() # backward
            optim.step() # update
            
            # log loss
            if (k+1) % print_every == 0:
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss-r {:8.3f} | loss-d {:8.3f}|' 
                    .format(k+1, elapsed*1000/print_every, loss_r.item(), loss_d.item())) 
                start_time = time.time()
            
            
    torch.save(vf.state_dict(), 'ckpt.pt')