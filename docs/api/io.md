# I/O

Mesh and point-cloud loading, saving, and diagnostic exports. All geometry I/O goes through Trimesh; diagnostic exports produce OBJ line-segment files for visual inspection in Rhino or other 3D viewers.

- **Mesh I/O** -- `load_mesh` reads PLY files and ensures consistent face-normal orientation; `save_mesh` writes PLY.
- **Point clouds** -- `save_pointcloud` writes Nx3 arrays as PLY point clouds.
- **Diagnostic exports** -- sun vectors, arbitrary rays, surface points with normals, and bounding-box meshes, all written as OBJ or PLY for debugging and visual verification.

::: urbansolarcarver.io
