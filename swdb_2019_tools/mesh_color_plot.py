# This function will color a mesh object with colors in vtk 
from meshparty import trimesh_vtk

def color_mesh_itkwidgets(color_vec, mesh):
    poly_data = trimesh_vtk.trimesh_to_vtk(mesh.vertices, mesh.faces)
    vtk_color_vec = trimesh_vtk.numpy_to_vtk(color_vec)
    vtk_color_vec.SetName('colors')
    poly_data.GetPointData().SetScalars(vtk_color_vec)
    return poly_data