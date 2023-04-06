# API Reference

---

<u>_wanghanweibnds2015@gmail.com_</u>

[<font size=5 color=53868b>data_interface.data_interface</font>](#data_interface)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_interface.get_field</font>](#data_interface_get_field)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_interface.get_line_data</font>](#data_interface_get_line_data)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_interface.get_point_data</font>](#data_interface_get_point_data)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_interface.get_surface_data</font>](#data_interface_get_surface_data)  
[<font size=5 color=53868b>data_interface.data_openfoam</font>](#data_openfoam)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_openfoam.read_field_data</font>](#data_openfoam_read_field_data)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.data_openfoam.read_mesh</font>](#data_openfoam_read_mesh)  
[<font size=5 color=53868b>data_interface.mesh</font>](#mesh)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.mesh.add_grid</font>](#mesh_add_grid)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.mesh.is_2d_grid</font>](#mesh_is_2d_grid)  
[<font size=4 color=7ac5cd>&nbsp;&nbsp;&nbsp;&nbsp;data_interface.mesh.set_grid</font>](#mesh_set_grid)  

## data_interface.data_interface<span id="data_interface"></span>

### data_interface.data_interface.get_field( field, t=0.0, vector=False, direction='x' )<span id="data_interface_get_field"></span>   
> **Function:**  
> Return specified field at given time, both scalar field and vector field supported.  
>   
> **Args:**  
> >**field : string**  
> >Physics field to be obtained.
> 
> >**t : float, optional**  
> >Specified time, 0.0 if steady case.
> 
> >**vector : bool, optional**  
> >Vector/scalar field, scalar defaulted.
> 
> >**direction : string, optional**  
> >Define which component to be obtained if vector field, x-direction defaulted.

### data_interface.data_interface.get_line_data( field, x, y, z=0, t=0, method="nearest" )<span id="data_interface_get_line_data"></span>  
> **Function:**  
> Return interpolated field data along a line (or curve).
>   
> **Args:**
> >**field : string**  
> >Physics field to be obtained.  
>  
> >**x : array like**  
> >1-D arrays representing coordinates in x-direction.  
>  
> >**y : array like**  
> >1-D arrays representing coordinates in y-direction.  
> 
> >**z : array like, optional**  
> >1-D arrays representing coordinates in z-direction, if 3-D case.  
>   
> >**t : float, optional**  
> >Specified time, 0.0 if steady case.    
>   
> >**method : string {"nearest", "linear", "cubic"}, optional**  
> >Method of interpolation, "nearest" defaulted.  
> > - <font color="#1e90ff">nearest</font>  
> > Return the value at the data point closet to the point of interpolation.  
> > - <font color="#1e90ff">linear</font>  
> > Tessellate the input point set to N-D simplices, and interpolate linearly on each simplex.  
> > - <font color="#1e90ff">cubic</font>  
> > Return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface.

### data_interface.data_interface.get_point_data( field, x, y, z=0, t=0, method="nearest" )<span id="data_interface_get_point_data"></span>  
> **Function:**  
> Return interpolated field data at specified point.
>   
> **Args:**
> >**field : string**  
> >Physics field to be obtained.  
>  
> >**x : array like**  
> >1-D arrays representing coordinates in x-direction.  
>  
> >**y : array like**  
> >1-D arrays representing coordinates in y-direction.  
> 
> >**z : array like, optional**  
> >1-D arrays representing coordinates in z-direction, if 3-D case.  
>   
> >**t : float, optional**  
> >Specified time, 0.0 if steady case.    
>   
> >**method : string {"nearest", "linear", "cubic"}, optional**  
> >Method of interpolation, "nearest" defaulted.  
> > - <font color="#1e90ff">nearest</font>  
> > Return the value at the data point closet to the point of interpolation.  
> > - <font color="#1e90ff">linear</font>  
> > Tessellate the input point set to N-D simplices, and interpolate linearly on each simplex.  
> > - <font color="#1e90ff">cubic</font>  
> > Return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface.

### data_interface.data_interface.get_surface_data( field, x, y, z=0, t=0, method="nearest" )<span id="data_interface_get_surface_data"></span>  
> **Function:**  
> Return interpolated field data in a surface.
>   
> **Args:**
> >**field : string**  
> >Physics field to be obtained.  
>  
> >**x : array like**  
> >1-D arrays representing coordinates in x-direction.  
>  
> >**y : array like**  
> >1-D arrays representing coordinates in y-direction.  
> 
> >**z : array like, optional**  
> >1-D arrays representing coordinates in z-direction, if 3-D case.  
>   
> >**t : float, optional**  
> >Specified time, 0.0 if steady case.    
>   
> >**method : string {"nearest", "linear"}, optional**  
> >Method of interpolation, "nearest" defaulted.  
> > - <font color="#1e90ff">nearest</font>  
> > Return the value at the data point closet to the point of interpolation.  
> > - <font color="#1e90ff">linear</font>  
> > Tessellate the input point set to N-D simplices, and interpolate linearly on each simplex.

## data_interface.data_openfoam<span id="data_openfoam"></span>

### data_interface.data_openfoam.read_field_data( *args )<span id="data_openfoam_read_field_data"></span>
> **Function:**  
> Read given field data from OpenFOAM projects.  
>   
> **Args:**
> >**\*args : strings**  
> >Physical field to read, multiple fields to be give at the same time is allowed.

### data_interface.data_openfoam.read_mesh( )<span id="data_openfoam_read_mesh"></span>
> **Function:**  
> Read mesh from OpenFOAM projects.

## data_interface.mesh<span id="mesh"></span>

### data_interface.mesh.add_grid( *args )<span id="mesh_add_grid"></span>  
> **Functions:**  
> Add grid information when grid already exists.
> 
> **Args:**  
> >**\*args : array like**  
> >grid matrix, 'x','y','z' coordinates expected to be listed in 1<sup>st</sup>, 2<sup>nd</sup>, 3<sup>rd</sup> column respectively (`numpy.array`, `numpy.matrix`, `list` supported).

### data_interface.mesh.is_2d_grid( *args )<span id="mesh_is_2d_grid"></span>
> **Function:**  
> Estimate if input is a 2-D grid or 3-D grid. Return 'True' if given grid is 2-D, 'False' if given grid is 3-D, otherwise, raise an error.  
>   
> **Args:** 
> >**\*args : array like**  
> >grid matrix, 'x','y','z' coordinates expected to be listed in 1<sup>st</sup>, 2<sup>nd</sup>, 3<sup>rd</sup> column respectively (`numpy.array`, `numpy.matrix`, `list` supported).

### data_interface.mesh.set_grid( *args )<span id="mesh_set_grid"></span>
> **Functions:**  
> Add grid information when no grid exists, overwrite existed grid.  
> 
> **Args:**
> >**\*args : array like**  
> >grid matrix, 'x','y','z' coordinates expected to be listed in 1<sup>st</sup>, 2<sup>nd</sup>, 3<sup>rd</sup> column respectively (`numpy.array`, `numpy.matrix`, `list` supported).

