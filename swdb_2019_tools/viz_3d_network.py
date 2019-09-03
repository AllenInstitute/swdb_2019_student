from meshparty import skeleton
from meshparty import trimesh_vtk
import pandas as pd
import numpy as np


def viz_3d_network(positions, edgelist):
    """
    Makes an interactive, 3d visualization of a spatial network.

    Parameters
    ----------
    positions: n_nodes x 3
        The 3d coordinates of each node in the network.

    edgelist: n_edges x 2
        List of the edges. If edges are labeled by node
        names (as opposed to node indexes) then positions
        must be a pd.DataFrame indexed by the node names.
    """

    # if nodes are named then convert node names to indexes
    if isinstance(positions, pd.DataFrame) and \
            isinstance(edgelist, pd.DataFrame):

        # convert edges to node index format i.e. (idx_0, idx_9) -> (0, 9)
        rootid2index = {idx: i for i, idx in enumerate(positions.index)}
        to_idxer = np.vectorize(lambda x: rootid2index[x])
        edgelist = to_idxer(edgelist)

    positions = np.array(positions)
    edgelist = np.array(edgelist)

    sk = skeleton.Skeleton(positions, edgelist)
    edge_actor = trimesh_vtk.skeleton_actor(sk,
                                            color=(0, 0, 0),
                                            opacity=.7,
                                            line_width=1)

    vert_actor = trimesh_vtk.point_cloud_actor(positions,
                                               size=3000,
                                               color=(1, 0, 0))
    trimesh_vtk.render_actors([edge_actor, vert_actor])
