

from abc import ABC, abstractmethod
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#from helper_tools import remove_duplicate_plot_descriptors


"""
AxesComponent Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AxesComponent(ABC):
    
    @abstractmethod
    def draw(self, ax: Axes, **kwargs):
        pass
    


"""
AxesComponentDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AxesComponentDecorator(AxesComponent):

    def __init__(self, axis_component: AxesComponent):
      
        self._axis_component = axis_component


    def draw(self, ax: Axes, **kwargs):
        
        self._axis_component.draw(ax=ax, **kwargs)



"""
ArtistAxesDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ArtistAxesDecorator(AxesComponent):

    def __init__(self, component: AxesComponent, artist: Artist):

        self.component = component
        self.artist = artist


    def draw(self, ax, **kwargs):

        self.component.draw(ax, **kwargs)
        
        ax.add_artist(self.artist)

  