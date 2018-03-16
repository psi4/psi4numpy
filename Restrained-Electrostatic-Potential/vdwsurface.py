import numpy as np

#Van der Waals radii (in angstrom) are taken from GAMESS.
vdw_r = {'H': 1.20, 'HE': 1.20,
         'LI': 1.37, 'BE': 1.45, 'B': 1.45, 'C': 1.50,
         'N': 1.50, 'O': 1.40, 'F': 1.35, 'NE': 1.30,
         'NA': 1.57, 'MG': 1.36, 'AL': 1.24, 'SI': 1.17,
         'P': 1.80, 'S': 1.75, 'CL': 1.70}

def surface(n):
    """Compute approximately n points on unit sphere
       Adapted from GAMESS
    input:
        n: integer: approximate number of requested surface points
    output:
        np.array(u): nupmy array of xyz coordinates of surface points
    """
        
    u = []
    eps = 1e-10
    nequat = int(np.sqrt(np.pi*n))
    nvert = int(nequat/2)
    nu = 0
    for i in range(nvert+1):
        fi = np.pi*i/nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat*xy+eps)
        if nhor < 1:
            nhor = 1
        for j in range(nhor):
            fj = 2*np.pi*j/nhor
            x = np.cos(fj)*xy
            y = np.sin(fj)*xy
            if nu >= n:
                return np.array(u)
            nu += 1
            u.append([x, y, z])
    return np.array(u) 

def vdw_surface(coordinates, elements, scale_factor, density, input_radii):
    """Compute points on the van der Wall surface of molecules

    input:
        coordinates: cartesian coordinates of the nuclei, in units of angstrom
        elements: list
            The symbols (e.g. C, H) for the nuceli
        scale_factor: float
            The points on the molecular surface are set at a distance of
            scale_factor * vdw_radius away from each of the atoms.
        density: float
            The (approximate) number of points to generate per square angstrom
            of surface area. 1.0 is the default recommended by Kollman & Singh.
        input_radii: dictionary of user's defined VDW radii

    output:
        radii: dictionary of scaled VDW radii
        surface_points: numpy array
   """
    radii = {}
    surface_points = []
    # scale radii
    for i in elements:
        if i in radii.keys():
            continue
        if i in input_radii.keys():
            radii[i] = input_radii[i] * scale_factor
        elif i in vdw_r.keys():
            radii[i] = vdw_r[i] * scale_factor
        else:
            raise KeyError('%s is not a supported element; ' %i
                         + 'use the "RADIUS" option to add '
                         + 'its van der Waals radius.')
    for i in range(len(coordinates)):
        # calculate points
        n_points = int(density * 4.0 * np.pi* np.power(radii[elements[i]], 2))
        dots = surface(n_points)
        dots = coordinates[i] + radii[elements[i]] * dots
        for j in range(len(dots)):
            save = True
            for k in range(len(coordinates)):
                if i == k:
                    continue
                # exclude points within the scaled VDW radius of other atoms
                d = np.linalg.norm(dots[j] - coordinates[k])
                if d < radii[elements[k]]:
                    save = False
                    break
            if save:
                surface_points.append(dots[j])
    return radii, np.array(surface_points)
