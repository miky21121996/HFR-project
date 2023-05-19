import sys, os
import time
import math
import numpy as np
import datetime as dt
from calendar import monthrange
import xarray as xr
from netCDF4 import Dataset, date2num, num2date
from scipy.spatial import Delaunay
import scipy.linalg as sc
import scipy.special as sp
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull

def putna(left, right, xar, scalar = None):
    '''
    Put NaN in xarray according if they are laying in the interval `left,right`

    Parameters
    ==========
    left, right:
        Extremes of the interval where the values must be NaN
    xar:
        xarray
    scalar :
        If set all entries not satisfying the condition are put equal to `scalar`

    Returns
    =======
    Modified array
    '''

    if scalar:
        out=scalar
    else:
        out=xar

    return xr.where((xar < right) & (xar > left), np.nan, out)

def get_sea(maskT):
    '''
    Obtain indexed coordinates for sea points.
    It works for 3D and 2D (fixed level) masks.
    '''

    # Try Interpolation
    tm = putna(-0.1, 0.1, maskT)
    sea_index = ~np.isnan(tm).stack(ind=maskT.dims)

    maskT_vec = tm.stack(ind=maskT.dims)
    land_point = maskT_vec[~sea_index]
    sea_point = maskT_vec[sea_index]
    land_point.name = 'Land'
    sea_point.name = 'Sea'

    # Compute triangularization for sea points (only if 2D)
    latlon = np.asarray([sea_point.lat.data, sea_point.lon.data]).T

    return latlon, sea_index, maskT_vec

class Ocean_Interpolator():
    """This class creates weights for interpolation of ocean fields.

    This class creates an interpolation operator from a *source grid*
    `src_grid` to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation. The model uses
    an Arakawa C-grid, that is shown in the following picture. The f-points
    correspond to the points where the Coriolis terms are carried.

    .. image:: ../resources/NEMOgrid.png
        :scale: 25 %
        :align: right

    The Arakawa C-grid used in the ocean model shows also the ordering
    of the points, indicating which points correspond to the (i,j) index.

    The *source grid* must be a `xarray` `DataSet` containing coordinates
    `latitude` and `longitude`.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Border land points at all levels are covered by a convolution value using a
    window that can be changed in `sea_over_land`.

    Works only on single `DataArray`

    Parameters
    ----------

    src_grid : xarray
        Source grid
    tgt_grid : xarray
        Target grid
    level : float
        Depth to generate the interpolator
    window : int
        Window for sea over land
    period : int
        Minimum number of points in the sea-over-land process
    nloops : int
        Number of sea-over-land loops
    verbose: bool
        Lots of output

    Attributes
    ----------

    tgt_grid :
        Target grid
    mask :
        Mask of the target grid
    vt :
        Weights
    wt :
        Weights
    mdir :
        Directory for masks files
    ingrid :
        Input Grid, `src_grid_name`
    outgrid :
        Output Grid `tgt_grid_name`
    window :
        Window for convolution Sea-Over-Land (default 3)
    period :
        Minimum number of points into the window (default 1)
    nloops :
        Number of sea-over-land loops
    T_lon :
        Longitudes of input T-mask
    T_lat :
        Latitudes of input T-mask
    U_lon :
        Longitudes of input U-mask
    U_lat :
        Latitudes of input U-mask
    V_lon :
        Longitudes of input V-mask
    V_lat :
        Latitudes of input V-mask
    tangle :
        Angles of the T points of the input grid
    mask_reg :
        Mask of the Target grid
    cent_long :
        Central Longitude of the Target grid
    name :
        Name of the Interpolator Object
    level :
        Level of the Interpolator Object


    Methods
    -------

    Interp_T :
        Interpolate Scalar quantities at T points

    Interp_UV :
        Interpolate Vector Velocities at (U,V)

    mask_sea_over_land :
        Mask border point for Sea over land

    fill_sea_over_land :
        Fill U,V values over land

    to_file :
        Writes interpolator object to file (pickled format)

    Examples
    --------
    Create the weights for interpolation

    >>> w = zint.Ocean_Interpolator(src_grid_name, tgt_grid_name)
    Interpolate temperature

    >>> target_xarray = w.interp_T(src_xarray, method='linear')

    Intek': grid.umask, 'vmask': grid.vmaskpolate U,V

    >>> target_xarray = w.interp_UV(U_xarray, V_xarray, method='linear')


    """

    __slots__ = ('mask','masku','maskv','mdir', 'mask_reg', \
                'sea_index','sea_index_U','sea_index_V', \
                'latlon', 'masT_vec', 'maskub','maskvb','masktb',\
                'latlon_reg','sea_index_reg','regmask_vec', \
                'T_lat','T_lon','U_lat','U_lon','V_lat','V_lon',\
                'name','cent_long','tri_sea_T','tri_sea_U','tri_sea_V','tangle',\
                    'ingrid','outgrid','level','window','period','nloops','empty','T_only',\
                    'Umask_reg','Ulatlon_reg','Usea_index_reg','Uregmask_vec',\
                    'Vmask_reg','Vlatlon_reg','Vsea_index_reg','Vregmask_vec',
                    'newSoL') #TODO newSoL is provisional -- for testing only

    def __init__(self, src_grid_name, tgt_grid_name,grid, verbose=False, window=3, period=1,
                 nloops=3, T_only=True, newSoL=True):  #TODO newSoL is provisional -- for testing only
        # Put here info on grids to be obtained from __call__
        # This currently works with mask files
        # 'masks_CMCC-CM2_VHR4_AGCM.nc'
        # and
        # 'ORCA025L50_mesh_mask.nc'

        # Path to auxiliary Ocean files
        homedir = os.path.expanduser("~")
        #self.mdir = homedir + '/Dropbox (CMCC)/data_zapata'
        self.mdir = '/work/oda/mg28621/prova_destag/tool_hfr/'
        self.ingrid = src_grid_name
        self.outgrid = tgt_grid_name

        # Parameter for sea over land
        self.window = window
        self.period = period
        self.nloops = nloops

        # Check levels #TODO why this?
        #if level > 0:
        #    self.level=level
        #else:
        #    SystemError(f'Surface variable not available, Level {level}')

        #Resolve grids
        s_in = self._resolve_grid(src_grid_name,grid)

        self.empty = False
        if s_in['tmask'].sum() == 0.:
            self.empty = True
            print(f'Level {level} is empty')
            return

        tk = s_in['tmask']
        self.T_lon = s_in['lonT']
        self.T_lat = s_in['latT']
        mask = tk.assign_coords({'lat':self.T_lat,'lon':self.T_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})
        #TODO add depth coordinate if level=None (also to U and V)

        if s_in['umask'] is not None:
           tk = s_in['umask']
           print("s_in umask shape: ",tk.shape)
           self.U_lon = s_in['lonU']
           self.U_lat = s_in['latU']
           masku = tk.assign_coords({'lat':self.U_lat,'lon':self.U_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        if s_in['vmask'] is not None:
           tk = s_in['vmask']
           self.V_lon = s_in['lonV']
           self.V_lat = s_in['latV']
           maskv = tk.assign_coords({'lat':self.V_lat,'lon':self.V_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        self.name = 'UV  Velocity'
        #self.level = str(level)

        #Fix Polar Fold
        #TODO add a flag ('global=True/False') to eventually do this fix
        #mask[-3:,:] = False

        #T angles
        self.tangle = s_in['tangle']

        #self.mask = xr.where(mask !=0, 1, np.nan)  ## this is the 1st step in mask_sea_over_land()
        #self.masktb, dum = self.mask_sea_over_land(mask)

        #TODO newSoL param is provisional -- only for testing
        if newSoL:
            self.mask = self.ALT_mask_sea_over_land(mask)
            if s_in['umask'] is not None:
                self.masku = self.ALT_mask_sea_over_land(masku)
            if s_in['vmask'] is not None:
                self.maskv = self.ALT_mask_sea_over_land(maskv)
        else:
            self.masktb, self.mask = self.mask_sea_over_land(mask)
            self.maskub, self.masku = self.mask_sea_over_land(masku)
            self.maskvb, self.maskv = self.mask_sea_over_land(maskv)

        print(f' Generating interpolator for {self.name}')
        if verbose:
            print(self.mask, self.masku, self.maskv)

        # Get triangulation for all grids
        self.latlon, self.sea_index, self.masT_vec = get_sea(self.mask)
        self.tri_sea_T  = Delaunay(self.latlon)  # Compute the triangulation for T
        print(f' computing the triangulation for T grid')

        if s_in['umask'] is not None:
            latlon_U, self.sea_index_U, masU_vec = get_sea(self.masku)
            self.tri_sea_U  = Delaunay(latlon_U)  # Compute the triangulation for U
            print(f' computing the triangulation for U grid')
        if s_in['vmask'] is not None:
            latlon_V, self.sea_index_V, masT_vec = get_sea(self.maskv)
            self.tri_sea_V  = Delaunay(latlon_V)  # Compute the triangulation for V
            print(f' computing the triangulation for V grid')

        # Target (T) Grid
        s_out = self._resolve_grid(tgt_grid_name,grid, verbose=verbose)
        self.mask_reg = s_out['mask']
        #self.cent_long = s_out['cent_long']
        self.latlon_reg, self.sea_index_reg, self.regmask_vec = get_sea(self.mask_reg)

        if not T_only:
            # Target (U) Grid
            self.Umask_reg = s_out['umask']
            self.Ulatlon_reg, self.Usea_index_reg, self.Uregmask_vec = get_sea(self.Umask_reg)
            # Target (V) Grid
            self.Vmask_reg = s_out['vmask']
            self.Vlatlon_reg, self.Vsea_index_reg, self.Vregmask_vec = get_sea(self.Vmask_reg)


    def __call__(self):
        print(f' Interpolator for T, U, V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        print(f' Main methods interp_T and interp_UV')
        return


    def __repr__(self):
        '''  Printing other info '''
        print(f' Interpolator for T, U, V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        return '\n'


    def interp_T(self, xdata, method='linear', grd='T'):
        '''
        Perform interpolation for T Grid point to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`

        method : str
            Method for interpolation
                * 'linear'  , Use linear interpolation
                * 'nearest' , use nearest interpolation

        grd : str
            can be either T (default), U or V, it defines the target grid

        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        '''
        regmask_vec = self.Uregmask_vec if grd=='U' else self.Vregmask_vec if grd=='V' else self.regmask_vec
        latlon_reg = self.Ulatlon_reg if grd=='U' else self.Vlatlon_reg if grd=='V' else self.latlon_reg
        sea_index_reg = self.Usea_index_reg if grd=='U' else self.Vsea_index_reg if grd=='V' else self.sea_index_reg

        # Fill T values over land
        xdata = self.xr_seaoverland(xdata)

        # stack the input and keep only the sea
        Tstack = xdata.stack(ind=('y','x'))
        sea_T = Tstack[self.sea_index]
        # prepare the (stacked) output
        temp = xr.full_like(regmask_vec, np.nan)

        # define the interpolator
        if method == 'linear':
            interpolator = spint.LinearNDInterpolator(self.tri_sea_T, sea_T)
        elif method == 'nearest':
            interpolator = spint.NearestNDInterpolator(self.tri_sea_T, sea_T)
        else:
            SystemError(f'Error in interp_T , wrong method  {method}')
        # Interpolate
        T_reg = interpolator(latlon_reg)
        # save the output
        temp[sea_index_reg] = T_reg.data
        out = temp.unstack()

        # Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            out[:,1439] = out[:,1438] + delx*(out[:,1]-out[:,1438])/ddelx
            out[:,0] = out[:,1438] + 2*delx*(out[:,1]-out[:,1438])/ddelx

        return out


    def interp_UV(self, udata, vdata, method = 'linear'):
        '''
        Perform interpolation for U,V Grid point to the target grid.
        This methods can be used for vector quantities.

        The present method interpolates the U,V points to the T points,
        rotates them and then interpolates to the target grid.

        Parameters
        ----------
        udata,vdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`

        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        '''

        # Insert NaN
        udata = xr.where(udata < 200, udata, np.nan)
        vdata = xr.where(vdata < 200, vdata, np.nan)

        # Fill U, V values over land
        udata = self.xr_seaoverland(udata)
        vdata = self.xr_seaoverland(vdata)

        # Compute interpolation for U,V
        Ustack = udata.stack(ind=('y','x'))
        Vstack = vdata.stack(ind=('y','x'))
        print(udata.shape)
        print(self.masku.shape)
        sea_U = Ustack[self.sea_index_U]
        sea_V = Vstack[self.sea_index_V]

        # Interpolate to T_grid
        if method == 'linear':
            int_U = spint.LinearNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.LinearNDInterpolator(self.tri_sea_V, sea_V)
        elif method == 'nearest':
            int_U = spint.NearestNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.NearestNDInterpolator(self.tri_sea_V, sea_V)
        else:
            SystemError(f'Error in interp_UV , wrong method  {method}')

        U_on_T = int_U(self.latlon)
        V_on_T = int_V(self.latlon)

        UT = self.masT_vec.copy()
        VT = self.masT_vec.copy()
        UT[self.sea_index] = U_on_T.data
        VT[self.sea_index] = V_on_T.data

        UT = UT.unstack()
        VT = VT.unstack()

        # Rotate Velocities
        fac = self.tangle*math.pi/180.
        print("UT: ",UT)
        print("fac: ",fac)
        uu = (UT * np.cos(fac) - VT * np.sin(fac)).drop_vars(['lon','lat'])
        vv = (VT * np.cos(fac) + UT * np.sin(fac)).drop_vars(['lon','lat'])

        # Interpolate to regular grid
        Uf = self.interp_T(uu, method=method, grd='T')
        Vf = self.interp_T(vv, method=method, grd='T')
        # Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            Uf[:,1439] = Uf[:,1438] + delx*(Uf[:,1]-Uf[:,1438])/ddelx
            Uf[:,0] = Uf[:,1438] + 2*delx*(Uf[:,1]-Uf[:,1438])/ddelx

        return Uf,Vf,UT,VT


    def to_file(self, filename):
        '''
        This method writes to file the interpolating object in
        pickled format
        '''

        with gzip.open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def mask_sea_over_land(self, mask):
        '''
        Mask border point for `Sea over land`.

        Sea over land is obtained by forward and backward filling NaN land
        value with an arbitrary value, then they are masked to reveal only the
        coastal points.

        Parameters
        ==========
        mask:
            original mask

        Returns
        =======
        border:
            border mask
        '''
        # np.nan where the mask=0, 1 elsewhere
        masknan = xr.where(mask != 0, 1, np.nan)
        # propagate in the x-direction (keep only the x-border)
        um1 = masknan.ffill(dim='x',limit=1).fillna(0.) + masknan.bfill(dim='x',limit=1).fillna(0.)- 2*masknan.fillna(0)
        # propagate in the y-direction (keep only the y-border)
        um2 = masknan.ffill(dim='y',limit=1).fillna(0.) + masknan.bfill(dim='y',limit=1).fillna(0.)- 2*masknan.fillna(0)
        # keep only the border
        bord = (um1+um2)/2
        # add the inside
        um = bord + masknan.fillna(0)
        # uniform the non-zero values (border + inside) to all 1s
        um = xr.where(um!=0, 1, np.nan)
        # border mask: 1 for border points, np.nan elsewhere
        mb = xr.where(bord !=0, 1, np.nan)

        return mb, um


    def ALT_mask_sea_over_land(self, mask):
        '''
        '''
        # np.nan where the mask=0, 1 elsewhere
        masknan = xr.where(mask != 0, 1, np.nan)
        # number of iterations of seaoverland
        for loop in range(self.nloops):
            # initialize the tuple to store the shifts
            mask_shift = ()
            # shift in all 8 directions
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if ((x != 0) | (y != 0)): # skip the no-shifting
                        # store the shifting in the tuple
                        mask_shift = mask_shift + (masknan.shift({'x':x, 'y':y}),)
            # take the max over 'shift' (either 1 or np.nan)
            masknan = xr.concat(mask_shift, dim='shift').max(dim='shift')

        return masknan
    def _resolve_grid(self, ingrid, grid,verbose=False):
        '''
        Internal routine to resolve grid informations
        '''
        print("ingrid: ",ingrid)

        if ingrid == 'eNEATL36':
            print(f' Tripolar 1/36 Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + 'eNEAT36_TWIN_coordinates.nc')
            #grid = grid.sel(z=level)
            #geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            angle = xr.open_dataset(self.mdir + '/eNEAT36_TWIN_angle.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': grid.glamt,'latT':grid.gphit,'lonU':grid.glamu, \
                    'latU':grid.gphiu,'lonV':grid.glamv,'latV':grid.gphiv  }
        if ingrid == '1_AGRIF':
            print(f' Tripolar 1/128 Grid -- {ingrid}')
            grid_model = xr.open_dataset(self.mdir + '1_BIZoo_coordinates.nc')
            #grid = grid.sel(z=level)
            #geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            angle = xr.open_dataset(self.mdir + '/1_BIZoo_angle.nc')
            struct={'tmask': grid_model.tmask, 'umask': grid_model.umask,'vmask': grid_model.vmask, 'tangle': angle.tangle, \
                    'lonT': grid_model.glamt,'latT':grid_model.gphit,'lonU':grid_model.glamu, \
                    'latU':grid_model.gphiu,'lonV':grid_model.glamv,'latV':grid_model.gphiv  }

        elif ingrid == 'GL_TV_HF_HFR-Gibraltar-Total':
            print(f' Regular HFR Lat-Lon Grid -- {ingrid}')
            #grid = xr.open_dataset('/work/oda/mg28621/prova_destag/tool_hfr/GL_TV_HF_HFR-Gibraltar-Total_coordinates.nc')
            #grid = grid.isel(z=level)
            cent_long = 720
            tk=np.logical_not(grid['mask'])
            mask = tk.assign_coords({'lat':grid.nav_lat,'lon':grid.nav_lon})
            struct={'mask': mask}
        elif ingrid == 'GL_TV_HF_HFR-TirLig-Total':
            print(f' Regular HFR Lat-Lon Grid -- {ingrid}')
            #grid = xr.open_dataset('/work/oda/mg28621/prova_destag/tool_hfr/GL_TV_HF_HFR-TirLig-Total_coordinates.nc'
            #grid = grid.isel(z=level)
            cent_long = 720
            #tk=np.logical_not(grid['mask'].data)
            tk=np.logical_not(grid['mask'])
            mask = tk.assign_coords({'lat':grid.nav_lat,'lon':grid.nav_lon})
            struct={'mask': mask}
        elif ingrid == 'GL_TV_HF_HFR-NAdr-Total':
            print(f' Regular HFR Lat-Lon Grid -- {ingrid}')
            #grid = xr.open_dataset('/work/oda/mg28621/prova_destag/tool_hfr/GL_TV_HF_HFR-NAdr-Total_coordinates.nc')
            #grid = grid.isel(z=level)
            cent_long = 720
            tk=np.logical_not(grid['mask'])
            mask = tk.assign_coords({'lat':grid.nav_lat,'lon':grid.nav_lon})
            struct={'mask': mask}
        elif ingrid == 'GL_TV_HF_HFR-Ibiza-Total':
            print(f' Regular HFR Lat-Lon Grid -- {ingrid}')
            #grid = xr.open_dataset('/work/oda/mg28621/prova_destag/tool_hfr/GL_TV_HF_HFR-Ibiza-Total_coordinates.nc')
            #grid = grid.isel(z=level)
            cent_long = 720
            tk=np.logical_not(grid['mask'])
            mask = tk.assign_coords({'lat':grid.nav_lat,'lon':grid.nav_lon})
            struct={'mask': mask}
        elif ingrid == 'GL_TV_HF_HFR-DeltaEbro-Total':
            print(f' Regular HFR Lat-Lon Grid -- {ingrid}')
            #grid = xr.open_dataset('/work/oda/mg28621/prova_destag/tool_hfr/GL_TV_HF_HFR-DeltaEbro-Total_coordinates.nc')
            #grid = grid.isel(z=level)
            cent_long = 720
            tk=np.logical_not(grid['mask'])
            mask = tk.assign_coords({'lat':grid.nav_lat,'lon':grid.nav_lon})
            struct={'mask': mask}
            
        elif ingrid == 'L44_025_TRP_GLO':
            print(f' Tripolar L44 0.25 Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L44_025_TRP_GLO/tmask44_UVT_latlon_coordinates.nc')
            grid = grid.sel(z=level)
            angle = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/ORCA025L75_angle.nc')
            geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': geo.glamt,'latT':geo.gphit,'lonU':geo.glamu, \
                    'latU':geo.gphiu,'lonV':geo.glamv,'latV':geo.gphiv  }

        elif ingrid == 'L50_025_TRP_GLO':
            print(f' Tripolar L50 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/mesh_mask.nc')
            # Take only a slice of it
            grid = grid.isel(x=slice(1000,1350),y=slice(600,750))
            grid = grid.isel(z=level, t=0)
            # remove the Red Sea
            grid = grid.where(np.logical_or(grid.y > 27, grid.x < 280), 0.)
            # remove the Black Sea
            grid = grid.where(np.logical_or(grid.y < 83, grid.x < 257), 0.)
            angle = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/ORCA025L75_angle.nc')\
                        .isel(x=slice(1000,1350),y=slice(629,779))
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask,
                    'tangle': angle.tangle, 'lonT': grid.glamt,'latT': grid.gphit,'lonU':grid.glamu,
                    'latU': grid.gphiu,'lonV': grid.glamv,'latV': grid.gphiv  }

        elif ingrid == 'L50_1o24_BDY_MED':
            print(f' LOBC L50 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/Med_LOBC/grids/tmask50_UVT_latlon_coordinates.nc').rename({'T_lat':'lat','T_lon':'lon'})
            # take only the boundary (T)
            nc_bndT_tmp = xr.open_dataset('/users_home/oda/pm28621/Med_LOBC/grids/bndT_2D.nc')
            # this Tgrid implies a masking
            #Tgrid = grid['tmask'].isel(x=nc_bndT_tmp.nbidta-1, y=nc_bndT_tmp.nbjdta-1) == 1
            #Tgrid = Tgrid.isel(z=level)
            # this one does not
            Tgrid = grid['lat'].isel(x=nc_bndT_tmp.nbidta-1, y=nc_bndT_tmp.nbjdta-1) > 0.
            # take only the boundary (U)
            nc_bndU_tmp = xr.open_dataset('/users_home/oda/pm28621/Med_LOBC/grids/bndU_2D.nc')
            #Ugrid = grid['umask'].isel(x=nc_bndU_tmp.nbidta-1, y=nc_bndU_tmp.nbjdta-1) == 1
            #Ugrid = Ugrid.isel(z=level)
            Ugrid = grid['lat'].isel(x=nc_bndU_tmp.nbidta-1, y=nc_bndU_tmp.nbjdta-1) > 0.
            # take only the boundary (V)
            nc_bndV_tmp = xr.open_dataset('/users_home/oda/pm28621/Med_LOBC/grids/bndV_2D.nc')
            #Vgrid = grid['vmask'].isel(x=nc_bndV_tmp.nbidta-1, y=nc_bndV_tmp.nbjdta-1) == 1
            #Vgrid = Vgrid.isel(z=level)
            Vgrid = grid['lat'].isel(x=nc_bndV_tmp.nbidta-1, y=nc_bndV_tmp.nbjdta-1) > 0.
            # prepare the struct
            cent_long = 720
            struct={'tmask': Tgrid, 'tangle': None, 'cent_long': cent_long,
                    'umask': Ugrid, 'vmask': Vgrid}

        elif ingrid == 'L50_1o24_REG_MED':
            print(f' Regular L50 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/Med_LOBC/grids/tmask50_UVT_latlon_coordinates.nc').rename({'T_lat':'lat','T_lon':'lon'})
            grid = grid.isel(z=level)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': grid.umask, 'vmask': grid.vmask}

        elif ingrid == 'L102_025_WOA':
            print(f' Regular L102 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/WOA_meshmask.nc')
            grid = grid.isel(x=slice(627,867), y=slice(460,560))
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }

        elif ingrid == 'L102_1o24_REG_MED':
            print(f' Regular L102 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask102_T_latlon_coordinates.nc').rename({'T_lat':'lat','T_lon':'lon'})
            grid = grid.isel(z=level)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': grid.umask, 'vmask': grid.vmask}

        elif ingrid == 'L65_025_SDN_NA_v2':
            print(f' Regular L65 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/SDN_NA_v2_meshmask.nc')
            grid = grid.isel(y=slice(25,85), x=slice(140,-1))
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }

        elif ingrid == 'L65_1o24_REG_MED':
            print(f' Regular L65 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask65_T_latlon_coordinates.nc').rename({'T_lat':'lat','T_lon':'lon'})
            grid = grid.isel(z=level)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': None, 'vmask': None}

        elif ingrid == 'L107_025_SDN_NA_v1':
            print(f' Regular L107 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/SDN_NA_v1_meshmask.nc')
            grid = grid.isel(y=slice(50,170), x=slice(240,-1))
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }

        elif ingrid == 'L92_0125_SDN_MED':
            print(f' Regular L92 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/SDN_MED_meshmask.nc')
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }

        elif ingrid == 'L92_1o24_REG_MED':
            print(f' Regular L92 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask92_T_latlon_coordinates.nc').rename({'T_lat':'lat','T_lon':'lon'})
            grid = grid.isel(z=level)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': None, 'vmask': None}

        elif ingrid == 'L75_025_REG_GLO':
            print(f' Regular L75 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L75_025_REG_GLO/GLO-MFC_001_025_mask_bathy.nc').sel(z=level). \
                rename({'longitude':'lon','latitude':'lat'})
            struct={'tmask': grid.mask, 'tangle': None, 'cent_long': 720}

        elif ingrid == 'L44_025_REG_GLO':
            print(f' Regular L44 0.25 Lat-Lon Grid from WOA -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/WOA/m025x025L44.nc')
            grid = grid.sel(depth=level)
            cent_long = 720
            struct={'tmask': grid.m025x025L44, 'tangle': None ,'cent_long' : cent_long}

        else:
             SystemError(f'Wrong Option in _resolve_grid --> {ingrid}')

        if verbose:
            print(f'Elements of {ingrid} extracted \n ')
            for i in struct.keys():
              print(f' {i} \n')

        return struct


    def fill_sea_over_land(self, U, u_mask):
        '''
        Put values Sea over land.

        Using the mask of the border points, the border points are filled
        with the convolution in 2D, using a window of width `window`.
        The `period` value is controlling the minimum number of points
        within the window that is necessary to yield a result.

        `period` and `window` are assigned as attributes of the interpolator.

        Parameters
        ==========
        U:
            Field to be treated
        u_mask:
            Mask for the field

        Returns
        =======
        U:
            Filled array
        '''
        # replace each entry with the average of a square window (3x3) centered at the value
        # if there is at least <period> (1) non-nan value in the window
        r = U.rolling(x=self.window, y=self.window, min_periods=self.period, center=True)
        r1 = r.mean()

        # select the border
        # maskub is the boundary mask returned by mask_sea_over_land
        border = ~np.isnan(self.maskub).drop_vars({'lat','lon'}).stack(ind=self.maskub.dims)
        # stack U and r1
        UU = U.stack(ind=U.dims)
        rs = r1.stack(ind=r1.dims)
        # replace the border of U (previously nans) with the border of r1
        UU[border] = rs[border]

        UUU = UU.unstack()
        UUU = UUU.assign_coords({'lon':u_mask.lon,'lat':u_mask.lat})

        return UUU
    
    def seaoverland(self, input_matrix, depth=1):  # depth is to select the number of consequential mask points to fill
        # depth loop
        for d in range(depth):
            if np.sum(input_matrix.mask) == 0:  # nothing to fill
                return input_matrix
            else:
                # Create a m x n x 8 3D matrix in which, third dimension fixed, the other dimensions
                #  contains values that are shifted in one of the 8 possible direction compared to the original matrix
                shift_matrix = ma.array(np.empty(shape=(input_matrix.shape[0], input_matrix.shape[1], 8)),
                                        mask=True, fill_value=1.e20, dtype=float)
                # up shift
                shift_matrix[: - 1, :, 0] = input_matrix[1:, :]
                # down shift
                shift_matrix[1:, :, 1] = input_matrix[0: - 1, :]
                # left shift
                shift_matrix[:, : - 1, 2] = input_matrix[:, 1:]
                # right shift
                shift_matrix[:, 1:, 3] = input_matrix[:, : - 1]
                # up-left shift
                shift_matrix[: - 1, : - 1, 4] = input_matrix[1:, 1:]
                # up-right shift
                shift_matrix[: - 1, 1:, 5] = input_matrix[1:, : - 1]
                # down-left shift
                shift_matrix[1:, : - 1, 6] = input_matrix[: - 1, 1:]
                # down-right shift
                shift_matrix[1:, 1:, 7] = input_matrix[: - 1, : - 1]
                # Mediate the shift matrix among the third dimension
                mean_matrix = ma.mean(shift_matrix, 2)
                # Replace input missing values with new ones belonging to the mean matrix
                input_matrix = ma.array(np.where(mean_matrix.mask + input_matrix.mask, mean_matrix, input_matrix),
                                        mask=mean_matrix.mask, fill_value=1.e20, dtype=float)
                input_matrix = ma.masked_where(mean_matrix.mask, input_matrix)
        return input_matrix


    def xr_seaoverland(self, var_in,
                       #nloops = 1, ## this is replaced by the 'nloops' parameter of the OceanInterpolator
                       xdim='x', ydim='y',
                       ismax = False):
        '''
        Fill nan values contained in var_in with the average (or max) value
        of the shifted versions of var_in

        Create a nD x 8 matrix in which, last dimension fixed, the other dimensions
        contain values that are shifted in one of the 8 possible directions
        of the last two axes compared to the original matrix
        '''
        var_out = var_in.copy()

        for loop in range(self.nloops):
            # Initialize the tuple to store the shifts
            var_shift = ()
            # Shift in all 8 directions
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if ((x != 0) | (y != 0)): # Skip the no-shifting
                        # Store the shifting in the tuple
                        var_shift = var_shift + (var_out.shift({xdim:x, ydim:y}),)
            # Take either the mean or the max over 'shift'
            if ismax:
                var_mean = xr.concat(var_shift, dim='shift').max(dim='shift')
            else:
                var_mean = xr.concat(var_shift, dim='shift').mean(dim='shift')
            # Replace input masked points (nan values) with new ones
            # from the mean (or max) matrix
            var_out = var_out.where(~np.isnan(var_out), other=var_mean)
            # Nothing more to flood
            if np.sum(np.isnan(var_out)) == 0:
                print('WARNING. Field does not have anymore land points,', str(loop + 1),
                      'steps were sufficient to flood it completely.', file=sys.stderr)
                break

        return var_out
