import trimesh
import plotly.graph_objects as go
import numpy as np


class Viewer():
    """Interactive visualizer for meshes and points clouds.
    """
    def __init__(self) -> None:
        self.viewer = go.Figure()

    def add_mesh(self, mesh:trimesh.base.Trimesh, 
                 color:str = 'blue', 
                 opacity:float = 1.0) -> None:
        """Adds a mesh trace to the viewer.

        Args:
            mesh: Input mesh
            color: Color of the mesh.
            opacity: Value from 0 to 1, determines 
                     the opacity of the trace.
        """
        mesh_object = go.Mesh3d(x = mesh.vertices[:, 0],
                                y = mesh.vertices[:, 1],
                                z = mesh.vertices[:, 2],
                                i = mesh.faces[:, 0],
                                j = mesh.faces[:, 1],
                                k = mesh.faces[:, 2],
                                color = color,
                                opacity = opacity)
        self.viewer.add_trace(mesh_object)

    def add_pc(self, pc:np.ndarray,  
                 opacity:float = 1.0) -> None:
        """Adds a point cloud trace to the viewer.

        Args:
            pc: Input point cloud (x, y, z).
            opacity: Value from 0 to 1, determines 
                     the opacity of the trace.
        """
        pc_object = go.Scatter3d(x = pc[:, 0],
                                 y = pc[:, 1],
                                 z = pc[:, 2],
                                 mode = 'markers',
                                 opacity = opacity)
        self.viewer.add_trace(pc_object)
        
    def show(self) -> None:
        self.viewer.show()