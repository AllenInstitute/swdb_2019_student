import numpy as np
import pandas as pd
import vtk
from meshparty import trimesh_vtk
import os

def convert_to_nm(col, voxel_size=[4,4,40]):
    """useful function for converting a pandas data frame voxel position
    column to a np.array of Nx3 size in nm

    Parameters
    ----------
    col : pandas.Series or np.array
        column from a datalink dataframe query, containing lists of x,y,z coordinates in voxels
        len(col)=N entries
    voxel_size : length(3) iterator
        a list, tuple, or numpy array with the desired voxel size (default = [4,4,40])

    Returns
    -------
    np.array
        a Nx3 array of positions converted to nanometers

    Example
    -------
    df = dl.query_cell_types('soma_valence_v2')
    soma_pos_nm = convert_to_nm(df.pt_position)
    """
    return np.vstack(col.values)*voxel_size


def synapse_to_matrix(syn_df, aggfunc = np.sum, fillvalue=0):
    """function for converting a pandas dataframe of synapses to a connectivity matrix
    
    Parameters
    ----------
    syn_df : pd.DataFrame
        Dataframe of synapses pulled from query_synapses
    aggfunc : func
        function to aggregate all the synapse sizes between pairs
        (default = np.sum)
        examples) np.sum (total synapse size)
                  np.mean (avg synapse size)
                  len (number of synapses)
                  np.std (variation of synapses)
    fillvalue : Number
        what to fill in the matrix when there are no synapses (default 0)
    
    Returns
    -------
    pd.DataFrame
        a dataframe whose rows are indexed by root_id, and columns are indexed by root id,
        values are the aggregated quantification of synapses between those.

    Example
    -------
    syn_df = dl.query_synapses('pni_synapses_i3', pre_ids = [neuron_list], post_ids = [neuron_list])
    conn_df = synapse_to_matrix(syn_df)
    
    # plot the matrix as an image
    f, ax = plt.subplots()
    ax.imshow(conn_df.values)

    """
    pd.pivot_table(syn_df,
                   values = 'size',
                   index = 'pre_pt_root_id',
                   columns = 'post_pt_root_id',
                   aggfunc=len,
                   fill_value = 0)


def render_actors_360(actors, directory, nframes, camera_start =None, start_frame=0,
            video_width=1280, video_height=720, scale=4, do_save=True):
    """
    Function to create a series of png frames which rotates around
    the Azimuth angle of a starting camera

    This will save images as a series of png images in the directory
    specified.

    The movie will start at time 0 and will go to frame nframes,
    completing a 360 degree rotation in that many frames.
    Keep in mind that typical movies are encoded at 15-30
    frames per second and nframes is units of frames.
    
    Parameters
    ----------
    actors :  list of vtkActor's
        list of vtkActors to render
    directory : str
        folder to save images into
    nframes : int
        number of frames to render
    camera_start : vtk.Camera
        camera to start rotation, default=None will fit actors in scene
    start_frame : int
        number to save the first frame number as... (default 0)
        i.e. frames will start_frame = 5, first file would be 005.png
    video_width : int
        size of video in pixels
    video_height : int
        size of the video in pixels
    scale : int
        how much to expand the image
    do_save : bool
        whether to save the images to disk or just play interactively

    Returns
    -------
    vtkRenderer
        the renderer used to render
    endframe
        the last frame written

    Example
    -------
    ::

        from meshparty import trimesh_io, trimesh_vtk
        mm = trimesh_io.MeshMeta(disk_cache_path = 'meshes')
        mesh = mm.mesh(filename='mymesh.obj')
        mesh_actor = trimesh_vtk.mesh_actor(mesh)
        mesh_center = np.mean(mesh.vertices, axis=0)
        camera_start = trimesh_vtk.oriented_camera(mesh_center)
        
        render_actors_360([mesh_actor], 'movie', 360, camera_start=camera_start)
    """     
    if camera_start is None:
        frame_0_file = os.path.join(directory, "0000.png")
        ren = trimesh_vtk.render_actors(actors,
                                        do_save=True,
                                        filename=frame_0_file,
                                        VIDEO_WIDTH=video_width,
                                        VIDEO_HEIGHT=video_height,
                                        back_color=(1,1,1))
        camera_start = ren.GetActiveCamera()
    
    cameras =[]
    times = []
    for k,angle in enumerate(np.linspace(0,360,nframes)):
        angle_cam = vtk.vtkCamera()
        angle_cam.ShallowCopy(camera_start)
        angle_cam.Azimuth(angle)
        cameras.append(angle_cam)
        times.append(k)
        
    return vtk_movie(actors, directory,
                     times = times,
                     cameras=cameras,
                     video_height=video_height,
                     video_width=video_width,
                     scale=scale,
                     do_save=do_save,
                     start_frame=start_frame)

def render_movie(actors, directory, times, cameras,start_frame=0,
                 video_width=1280, video_height=720, scale=4,
                 do_save=True, back_color=(1,1,1)):
    """
    Function to create a series of png frames based upon a defining 
    a set of cameras at a set of times.

    This will save images as a series of png images in the directory
    specified.

    The movie will start at time 0 and will go to frame np.max(times)
    Reccomend to make times start at 0 and the length of the movie
    you want.  Keep in mind that typical movies are encoded at 15-30
    frames per second and times is units of frames.
    
    Parameters
    ----------
    actors :  list of vtkActor's
        list of vtkActors to render
    directory : str
        folder to save images into
    times : np.array
        array of K frame times to set the camera to
    cameras : list of vtkCamera's
        array of K vtkCamera objects. movie with have cameras[k]
        at times[k]. 
    start_frame : int
        number to save the first frame number as... (default 0)
        i.e. frames will start_frame = 5, first file would be 005.png
    video_width : int
        size of video in pixels
    video_height : int
        size of the video in pixels
    scale : int
        how much to expand the image
    do_save : bool
        whether to save the images to disk or just play interactively

    Returns
    -------
    vtkRenderer
        the renderer used to render
    endframe
        the last frame written

    Example
    -------
    ::

        from meshparty import trimesh_io, trimesh_vtk
        mm = trimesh_io.MeshMeta(disk_cache_path = 'meshes')
        mesh = mm.mesh(filename='mymesh.obj')
        mesh_actor = trimesh_vtk.mesh_actor(mesh)

        mesh_center = np.mean(mesh.vertices, axis=0)
        
        camera_start = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, 1))
        camera_180 = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, -1))

        times = np.array([0, 90, 180])
        cameras = [camera_start, camera_180, camera_start]

        vtk_movie([mesh_actor],
                'movie',
                times,
                cameras)
    """     

    camera_interp=vtk.vtkCameraInterpolator()
    assert(len(times)==len(cameras))
    for t,cam in zip(times,cameras):
        camera_interp.AddCamera(t,cam)

    camera = vtk.vtkCamera()
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(video_width, video_height)
    ren.SetBackground(*back_color)
    ren.UseFXAAOn()
    # ren.SetBackground( 1, 1, 1)
    ren.SetActiveCamera(camera)
    renWin.Render()
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a )

    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renWin)
    imageFilter.SetScale(scale)
    imageFilter.SetInputBufferTypeToRGB()
    imageFilter.ReadFrontBufferOff()
    imageFilter.Update()

    #Setup movie writer
    moviewriter = vtk.vtkPNGWriter()
    moviewriter.SetInputConnection(imageFilter.GetOutputPort())

    for i in np.arange(0,np.max(times)+1):
        camera_interp.InterpolateCamera(i,camera)
        ren.ResetCameraClippingRange()
        camera.ViewingRaysModified()
        renWin.Render()
        filename = os.path.join(directory,"%04d.png"%(i+start_frame))
        
        moviewriter.SetFileName(filename)
        if do_save:
            #Export a single frame
            renWin.OffScreenRenderingOn()
            w2if = vtk.vtkWindowToImageFilter()
            
            w2if.SetInput(renWin)
            w2if.Update()

            imageFilter.Modified()
            moviewriter.Write()
    
    renWin.Finalize()
    return renWin,i+start_frame

