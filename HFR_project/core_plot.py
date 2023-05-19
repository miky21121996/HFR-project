import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import rgb2hex, TwoSlopeNorm
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress, pearsonr
from windrose import WindroseAxes


def plot_hfr_wind_field(info, extent, min_hfr, min_model, max_hfr, max_model, x, y, speed_hfr, U, V, skip, skip_coords, date_str, ds, output_plot_folder):

    plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.axes(projection=ccrs.Mercator())# Map projection
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--') #adding grid lines
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    #plotting antennas
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    if info['antennas']:
        for antenna in info['antennas']:
            plt.plot(antenna['lon'], antenna['lat'], color=np.random.rand(3,), markeredgecolor=np.random.rand(3,), marker='o',transform=ccrs.Geodetic(),label=antenna['name'])#add point 
            if id in list(ds.attrs.keys()):
                if ds.id in ["GL_TV_HF_HFR-TirLig-Total","GL_TV_HF_HFR-NAdr-Total"]:
                    ax.legend(loc='lower left')
                else:
                    ax.legend()

    #personalized limts
    ax.set_extent(extent)

    # quiver plot: set vectors, colormaps, colorbars
    norm = colors.Normalize(vmin=min(min_hfr,min_model), vmax=max(max_hfr,max_model))
    ax.pcolor(x, y, speed_hfr, cmap='viridis', vmin=min(min_hfr,min_model), vmax=max(max_hfr,max_model),
            transform=cartopy.crs.PlateCarree())

    quiver_label = 0.21
    Q=ax.quiver(x[skip_coords], y[skip_coords], U[skip], V[skip],
            transform=cartopy.crs.PlateCarree(),scale=0.5, scale_units='inches')
    
    ax.quiverkey(Q, 0.8, 1.07, quiver_label, str(quiver_label) + " m/s",fontproperties={'weight': 'bold'},labelpos='E')

    # title and colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',norm=norm)
    a = np.random.random((10, 20))
    im_ratio = a.shape[0]/a.shape[1]
    ticks = np.linspace(min(min_hfr,min_model), max(max_hfr,max_model), 5, endpoint=True)
    cb=plt.colorbar(sm,ax=ax, orientation='vertical', pad=0.15, fraction=0.047*im_ratio,format=FormatStrFormatter('%.2f'),ticks=ticks)
    cb.set_label(label='velocity (m/s)',fontsize=30)
    cb.ax.tick_params(labelsize=20)

    plt.title(date_str+' | ' + ds.id + '\n surface current velocity (OBS)', pad=28,fontsize=15)
    figure_name = ds.id+'_surface_current_velocity_'+ date_str +'.png'
    plt.savefig(output_plot_folder+figure_name, dpi=300, bbox_inches = "tight")
    plt.close()

def plot_model_wind_field(info, extent, min_hfr, min_model, max_hfr, max_model, x, y, skip, skip_coords, masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, date_str, output_plot_folder,label_plot,title_substring,name_file_substring,ds,spatial_mean_model_ts_instant):

    plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='k')
    # Map projection
    ax = plt.axes(projection=ccrs.Mercator())
    # adding grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #plotting antennas
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    if info['antennas']:
        for antenna in info['antennas']:
            plt.plot(antenna['lon'], antenna['lat'], color=np.random.rand(3,),markeredgecolor=np.random.rand(3,), marker='o',transform=ccrs.Geodetic(),label=antenna['name'])
            if id in list(ds.attrs.keys()):
                if ds.id in ["GL_TV_HF_HFR-TirLig-Total","GL_TV_HF_HFR-NAdr-Total"]:
                    ax.legend(loc='lower left')
                else:
                    ax.legend()

    # personalized limts
    ax.set_extent(extent)
    # quiver plot: set vectors, colormaps, colorbars
    print("max: ", max(max_hfr,max_model))
    norm = colors.Normalize(vmin=min(min_hfr,min_model), vmax=max(max_hfr,max_model))
    ax.pcolor(x, y, masked_speed_interpolated, cmap='viridis', vmin=min(min_hfr,min_model),
            vmax=max(max_hfr,max_model), transform=cartopy.crs.PlateCarree())

    quiver_label = 0.21
    Q=ax.quiver(x[skip_coords], y[skip_coords], masked_u_interpolated[skip],
            masked_v_interpolated[skip], transform=cartopy.crs.PlateCarree(), scale=0.5, scale_units='inches')
    ax.quiverkey(Q, 0.8, 1.07, quiver_label, str(quiver_label) + " m/s",fontproperties={'weight': 'bold'},labelpos='E')

    # title and colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',norm=norm)
    a = np.random.random((10, 20))
    im_ratio = a.shape[0]/a.shape[1]
    ticks = np.linspace(min(min_hfr,min_model), max(max_hfr,max_model), 5, endpoint=True)
    cb=plt.colorbar(sm,ax=ax, orientation='vertical', pad=0.15, fraction=0.047*im_ratio,format=FormatStrFormatter('%.2f'),ticks=ticks)
    cb.set_label(label='velocity (m/s)',fontsize=30)
    cb.ax.tick_params(labelsize=20)

    plt.title(date_str+' | ' + ds.id + '\n'+ title_substring + '\n' + label_plot + ' (MODEL)', pad=28,fontsize=15)
    figure_name = ds.id+name_file_substring+ date_str +'.png'
    plt.savefig(output_plot_folder+figure_name, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_bias(info, extent, x, y, min_bias, max_bias, masked_speed_interpolated, speed_hfr, date_str, output_plot_folder,label_plot,title_substring,name_file_substring,ds):

    plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='k')
    # Map projection
    ax = plt.axes(projection=ccrs.Mercator())
    # adding grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #plotting antennas
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    if info['antennas']:
        for antenna in info['antennas']:
            plt.plot(antenna['lon'], antenna['lat'], color=np.random.rand(3,),markeredgecolor=np.random.rand(3,), marker='o',transform=ccrs.Geodetic(),label=antenna['name'])
            if id in list(ds.attrs.keys()):
                if ds.id in ["GL_TV_HF_HFR-TirLig-Total","GL_TV_HF_HFR-NAdr-Total"]:
                    ax.legend(loc='lower left')
                else:
                    ax.legend()
    # personalized limts
    ax.set_extent(extent)
    norm = TwoSlopeNorm(vmin=min_bias, vcenter=0, vmax=max_bias)
    ax.pcolor(x, y, masked_speed_interpolated-speed_hfr, norm=norm, cmap='RdBu_r', transform=cartopy.crs.PlateCarree())

    # title and colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',norm=norm)
    a = np.random.random((10, 20))
    im_ratio = a.shape[0]/a.shape[1]
    ticks = np.linspace(min_bias, max_bias, 5, endpoint=True)
    cb=plt.colorbar(sm,ax=ax, orientation='vertical', pad=0.15, fraction=0.047*im_ratio,format=FormatStrFormatter('%.2f'),ticks=ticks)
    cb.set_label(label='velocity bias (m/s)',fontsize=30)
    cb.ax.tick_params(labelsize=20)

    plt.title(date_str+' | ' + ds.id + '\n '+ title_substring + '\n' + label_plot, pad=28,fontsize=15)
    figure_name = ds.id+ name_file_substring + date_str +'.png'
    plt.savefig(output_plot_folder+figure_name, dpi=300, bbox_inches = "tight")
    plt.close()

def plot_rmsd(info, extent, x, y, min_rmsd, max_rmsd, masked_speed_interpolated, speed_hfr, date_str, output_plot_folder,label_plot,title_substring,name_file_substring,ds):

    plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='k')
    # Map projection
    ax = plt.axes(projection=ccrs.Mercator())
    # adding grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #plotting antennas
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    if info['antennas']:
        for antenna in info['antennas']:
            plt.plot(antenna['lon'], antenna['lat'], color=np.random.rand(3,),markeredgecolor=np.random.rand(3,), marker='o',transform=ccrs.Geodetic(),label=antenna['name'])
            if id in list(ds.attrs.keys()):
                if ds.id in ["GL_TV_HF_HFR-TirLig-Total","GL_TV_HF_HFR-NAdr-Total"]:
                    ax.legend(loc='lower left')
                else:
                    ax.legend()
    # personalized limts
    ax.set_extent(extent)
    # quiver plot: set vectors, colormaps, colorbars
    norm = colors.Normalize(vmin=min_rmsd, vmax=max_rmsd)
    ax.pcolor(x, y, np.sqrt((masked_speed_interpolated-speed_hfr)**2), cmap='viridis', vmin=min_rmsd, vmax=max_rmsd, transform=cartopy.crs.PlateCarree())

    # title and colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',norm=norm)
    a = np.random.random((10, 20))
    im_ratio = a.shape[0]/a.shape[1]
    ticks = np.linspace(min_rmsd, max_rmsd, 5, endpoint=True)
    cb=plt.colorbar(sm,ax=ax, orientation='vertical', pad=0.15, fraction=0.047*im_ratio,format=FormatStrFormatter('%.2f'),ticks=ticks)    
    cb.set_label(label='velocity rmsd (m/s)',fontsize=30)
    cb.ax.tick_params(labelsize=20)

    plt.title(date_str+' | ' + ds.id + '\n ' + title_substring + '\n' + label_plot, pad=28, fontsize=15)
    figure_name = ds.id+ name_file_substring + date_str +'.png'
    plt.savefig(output_plot_folder+figure_name, dpi=300, bbox_inches = "tight")
    plt.close()

def plot_mod_obs_ts_comparison(obs_ts, mod_ts, time_res_to_average, ds, date_in, date_fin, output_plot_folder,timerange,name_exp,title_substring,name_file_substring):
    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average + name_file_substring +'.png'
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title(title_substring+': '+ ds.id + '\n Period: '+ date_in + '-' + date_fin, fontsize=29)
    
    mean_vel_mod = round(np.nanmean(np.array(mod_ts)),2)
    print("timerange shape: ",timerange.shape)
    print("mod_ts shape: ", np.array(mod_ts).shape)
    plt.plot(timerange,np.array(mod_ts),label = name_exp + ' : '+str(mean_vel_mod)+' m/s', linewidth=2)
    mean_vel_obs = round(np.nanmean(np.array(obs_ts)),2)
    plt.plot(timerange,np.array(obs_ts),label = 'Observation : '+str(mean_vel_obs)+' m/s', linewidth=2)
    plt.grid()
    ax.tick_params(axis='both', labelsize=26)
    if time_res_to_average[1]=='D':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_to_average[1]=='M':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_to_average[1]=='Y':
        ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    plt.legend(prop={'size': 30}, framealpha=0.2)
    plt.savefig(output_plot_folder + plotname)
    plt.close()

    return mean_vel_mod,mean_vel_obs

def line_A(x, m_A, q_A):
    return (m_A*x+q_A)

def BIAS(data,obs):
    return  np.round((np.nanmean( data-obs)).data, 2)

def RMSE(data,obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)),2)

def ScatterIndex(data,obs):
    num=np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom=np.sum(obs**2)
    return np.round(np.sqrt((num/denom)),2)

def Normalized_std(data,obs):
    data_std=np.std(data)
    data_obs=np.std(obs)
    return np.round(data_std/data_obs,2)

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,clip_on=False,cmap='plasma',**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def scatterPlot(mod, obs, outname, name, n_stations, n_time, possible_markers, hfr_name, pos_colors, time_string,**kwargs):

    if np.isnan(obs).any() or np.isnan(mod).any():

        obs_no_nan = obs[~np.isnan(obs) & ~np.isnan(mod)]
        mod_no_nan = mod[~np.isnan(obs) & ~np.isnan(mod)]
        xy = np.vstack([obs_no_nan, mod_no_nan])
    else:
        xy = np.vstack([obs, mod])

    color_list = pos_colors
    #possible_markers=np.array(["o","^","s","P","*","D"])
    if n_stations==1:
        print("prima repeat: ",possible_markers)
        m=np.repeat(possible_markers,len(obs[~np.isnan(obs) & ~np.isnan(mod)]))
        c_prova = np.tile(np.arange(0,6*len(obs),6),1)
        
    if n_stations>1:
        m=np.array([])
        c_prova = np.tile(np.arange(0,6*n_time,6),n_stations)
        for stat_counter,not_nan_num in enumerate(kwargs['len_not_nan_values']):
            m_element=np.repeat(possible_markers[stat_counter],not_nan_num)
            m=np.concatenate([m,m_element])
            
        print("all m: ",m)

    if np.isnan(obs).any() or np.isnan(mod).any():
        x, y = obs_no_nan, mod_no_nan
    else:
        x, y = obs, mod

    color_list_seq = np.tile(color_list[:n_time],n_stations)
    classes = time_string
    markers_labels = hfr_name
    fig, ax = plt.subplots(figsize=(10,6))

    im = mscatter(x, y, ax=ax, m=m, c=c_prova[~np.isnan(obs) & ~np.isnan(mod)],s=15)
    marker_array=[]
    if n_stations==1:
        marker_array.append(mlines.Line2D([], [], color='blue', marker=possible_markers[0], linestyle='None', markersize=5, label=markers_labels))
    else:
        for mark,mark_label in zip(possible_markers,markers_labels):
            print('label: ',mark_label)
            marker_array.append(mlines.Line2D([], [], color='blue', marker=mark, linestyle='None', markersize=5, label=mark_label))

    legend_1=plt.legend(handles=im.legend_elements(num=n_time)[0], labels=classes, loc='right',prop={"size":9},bbox_to_anchor=(1.3, 0.5))
    plt.legend(handles=marker_array,loc='upper left',prop={"size":12})
    plt.gca().add_artist(legend_1)

    maxVal = np.nanmax((x, y))
    ax.set_ylim(0, maxVal)
    ax.set_xlim(0, maxVal)
    ax.set_aspect(1.0)
    ax.tick_params(axis='both', labelsize=12.5)

    bias = BIAS(y,x)
    corr, _ = pearsonr(x, y)
    rmse=RMSE(y,x)
    nstd=Normalized_std(y,x)
    si=ScatterIndex(y,x)
    slope,intercept, rvalue,pvalue,stderr=linregress(y,x)

    prova = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(prova, y)
    xseq = np.linspace(0, maxVal, num=100)
    ax.plot(xseq, a*xseq, 'r-')
    plt.text(0.001, 0.7, name, weight='bold',transform=plt.gcf().transFigure,fontsize=13)

    plt.text(0.01, 0.32, 'Entries: %s\n'
            'BIAS: %s m/s\n'
            'RMSD: %s m/s\n'
            'NSTD: %s\n'
            'SI: %s\n'
            'corr:%s\n'
            'Slope: %s\n'
            'STDerr: %s m/s'
            %(len(obs),bias,rmse,nstd,si,np.round(corr,2),
                np.round(a[0],2),np.round(stderr,2)),transform=plt.gcf().transFigure,fontsize=15)

    stat_array=[bias,rmse,si,np.round(corr,2),np.round(stderr,2),len(obs)]

    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=15, x=0.5, y=1.01)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=18)

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=18)


    ax.plot([0,maxVal],[0,maxVal],c='k',linestyle='-.')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig(outname)
    plt.close()
    return stat_array

def plot_mod_obs_ts_comparison(obs_ts, mod_ts, time_res_to_average, ds, date_in, date_fin, output_plot_folder,timerange,name_exp,title_substring,name_file_substring):
    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average + name_file_substring +'.png'
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title(title_substring+': '+ ds.id + '\n Period: '+ date_in + '-' + date_fin, fontsize=29)
    
    mean_vel_mod = round(np.nanmean(np.array(mod_ts)),2)
    print("timerange shape: ",timerange.shape)
    print("mod_ts shape: ", np.array(mod_ts).shape)
    plt.plot(timerange,np.array(mod_ts),label = name_exp + ' : '+str(mean_vel_mod)+' m/s', linewidth=2)
    mean_vel_obs = round(np.nanmean(np.array(obs_ts)),2)
    plt.plot(timerange,np.array(obs_ts),label = 'Observation : '+str(mean_vel_obs)+' m/s', linewidth=2)
    plt.grid()
    ax.tick_params(axis='both', labelsize=26)
    if time_res_to_average[1]=='D':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_to_average[1]=='M':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_to_average[1]=='Y':
        ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    plt.legend(prop={'size': 30}, framealpha=0.2)
    plt.savefig(output_plot_folder + plotname)
    plt.close()

    return mean_vel_mod,mean_vel_obs

def plot_mod_obs_ts_comparison_1(obs_ts, mod_ts, time_res_to_average, ds, date_in, date_fin, output_plot_folder,timerange,name_exp,title_substring,name_file_substring,num_exp):
    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average + name_file_substring +'.png'
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title(title_substring+': '+ ds.id + '\n Period: '+ date_in + '-' + date_fin, fontsize=29)

    for exp in range(num_exp):
        mean_vel_mod = round(np.nanmean(np.array(mod_ts[exp][ds.id])),2)
        print("timerange shape: ",timerange.shape)
        print("mod_ts shape: ", np.array(mod_ts[exp][ds.id]).shape)
        plt.plot(timerange,np.array(mod_ts[exp][ds.id]),label = name_exp[exp] + ' : '+str(mean_vel_mod)+' m/s', linewidth=2)

    mean_vel_obs = round(np.nanmean(np.array(obs_ts[ds.id])),2)
    plt.plot(timerange,np.array(obs_ts[ds.id]),label = 'Observation : '+str(mean_vel_obs)+' m/s', linewidth=2)
    plt.grid()
    ax.tick_params(axis='both', labelsize=26)
    if time_res_to_average[1]=='D':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_to_average[1]=='M':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_to_average[1]=='Y':
        ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_to_average[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    plt.legend(prop={'size': 30}, framealpha=0.2)
    plt.savefig(output_plot_folder + plotname)
    plt.close()
    
def plot_interpolated_hfr_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x_subset_model, y_subset_model, skip, skip_coords, masked_hfr_speed_interpolated, masked_hfr_u_interpolated, masked_hfr_v_interpolated, date_str, output_plot_folder,title_substring,name_file_substring,ds):

    plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='k')
    # Map projection
    ax = plt.axes(projection=ccrs.Mercator())
    # adding grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #plotting antennas
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    if info['antennas']:
        for antenna in info['antennas']:
            plt.plot(antenna['lon'], antenna['lat'], color=np.random.rand(3,),markeredgecolor=np.random.rand(3,), marker='o',transform=ccrs.Geodetic(),label=antenna['name'])
    if id in list(ds.attrs.keys()):
        if ds.id in ["GL_TV_HF_HFR-TirLig-Total","GL_TV_HF_HFR-NAdr-Total"]:
            ax.legend(loc='lower left')
        else:
            ax.legend()
    # personalized limts
    ax.set_extent(extent)
    # quiver plot: set vectors, colormaps, colorbars
    print("max: ", max(max_hfr,max_model_value))
    norm = colors.Normalize(vmin=min(min_hfr,min_model_value), vmax=max(max_hfr,max_model_value))
    ax.pcolor(x_subset_model, y_subset_model, masked_hfr_speed_interpolated, cmap='viridis', vmin=min(min_hfr,min_model_value),
            vmax=max(max_hfr,max_model_value), transform=cartopy.crs.PlateCarree())
    # quiver plot: arrows
    Q=ax.quiver(x_subset_model[skip_coords], y_subset_model[skip_coords], masked_hfr_u_interpolated[skip]/max(max_hfr,max_model_value),
            masked_hfr_v_interpolated[skip]/max(max_hfr,max_model_value), transform=cartopy.crs.PlateCarree())
    quiver_label = np.round(0.5*(max(max_hfr,max_model_value)-min(min_hfr,min_model_value)),2)
    ax.quiverkey(Q, 0.1, 0.9, quiver_label, str(quiver_label) + " m/s",fontproperties={'weight': 'bold'})


    # title and colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',norm=norm)
    a = np.random.random((10, 20))
    im_ratio = a.shape[0]/a.shape[1]
    ticks = np.linspace(min(min_hfr,min_model_value), max(max_hfr,max_model_value), 5, endpoint=True)
    cb=plt.colorbar(sm,ax=ax, orientation='vertical', pad=0.15, fraction=0.047*im_ratio,format=FormatStrFormatter('%.2f'),ticks=ticks)
    cb.set_label(label='velocity (m/s)',fontsize=30)
    cb.ax.tick_params(labelsize=20)

    plt.title(date_str+' | ' + ds.id + '\n' + title_substring, pad=28,fontsize=15)
    figure_name = ds.id+name_file_substring+ date_str +'.png'
    plt.savefig(output_plot_folder+figure_name, dpi=300, bbox_inches = "tight")
    plt.close()
    
def plot_windrose(direction,velocity,minimum,maximum,ds,date_in,date_fin,name_file_substring,title_substring,output_plot_folder,ymin,ymax):
    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + name_file_substring +'.png'
    fig = plt.figure()
    #plt.rc('font', size=24)
    rect = [0.1, 0.1, 0.8, 0.8]
    hist_ax = plt.Axes(fig,rect)
    hist_ax.bar(np.array([1]), np.array([1]))
    ax = WindroseAxes.from_ax()
    turbo = plt.get_cmap('turbo')
    ax.bar(direction, velocity, normed=True, bins=np.linspace(minimum,maximum,5),opening=0.8,edgecolor='white',cmap=turbo)
    # set the y-axis tick positions
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_yticklabels(['{:.2f}'.format(x) for x in np.linspace(ymin, ymax, 5)], fontsize=12)
    # Set the y-axis tick format to percentage
    #ax.yaxis.set_major_formatter(plt.PercentFormatter())
    # get y-axis limits
    #ymin, ymax = ax.get_ylim()

    # print y-axis limits
    #print("y-axis limits:", ymin, ymax)
    
    ax.set_title(title_substring+':' + '\n Period: '+ date_in + '-' + date_fin, fontsize=12)
    #ax.set_legend(loc=4, bbox_to_anchor=(1.,-0.07))
    legend = ax.set_legend(loc=4, bbox_to_anchor=(1., -0.07), prop={'size': 12})

    # Increase the font size of the legend values
    for text in legend.get_texts():
        text.set_fontsize(14)
    plt.savefig(output_plot_folder + plotname,bbox_inches=None)
    plt.close()
    
class TaylorDiagram(object):
  def __init__(self, STD ,fig=None, rect=111, label='_'):
    self.STD = STD
    tr = PolarAxes.PolarTransform()
    # Correlation labels
    rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
    tlocs = np.arccos(rlocs) # Conversion to polar angles
    gl1 = gf.FixedLocator(tlocs) # Positions
    tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
    # Standard deviation axis extent
    self.smin = 0
    self.smax = 1.6 * self.STD[0]
    gh = fa.GridHelperCurveLinear(tr,extremes=(0,(np.pi/2),self.smin,self.smax),grid_locator1=gl1,tick_formatter1=tf1,)
    if fig is None:
        fig = plt.figure()
    ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
    fig.add_subplot(ax)
    # Angle axis
    ax.axis['top'].set_axis_direction('bottom')
    ax.axis['top'].label.set_text("Correlation coefficient")
    ax.axis['top'].label.set_size(32)
    ax.axis['top'].major_ticklabels.set_fontsize(16)
    ax.axis['top'].toggle(ticklabels=True, label=True)
    ax.axis['top'].major_ticklabels.set_axis_direction('top')
    ax.axis['top'].label.set_axis_direction('top')
    # X axis
    ax.axis['left'].set_axis_direction('bottom')
    ax.axis['left'].label.set_text("Standard deviation")
    ax.axis['left'].label.set_size(32)
    ax.axis['left'].major_ticklabels.set_fontsize(16)
    ax.axis['left'].toggle(ticklabels=True, label=True)
    ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
    ax.axis['left'].label.set_axis_direction('bottom')
    # Y axis
    ax.axis['right'].set_axis_direction('top')
    ax.axis['right'].label.set_text("Standard deviation")
    ax.axis['right'].major_ticklabels.set_fontsize(16)
    ax.axis['right'].label.set_size(32)
    ax.axis['right'].toggle(ticklabels=True, label=True)
    ax.axis['right'].major_ticklabels.set_axis_direction('left')
    ax.axis['right'].label.set_axis_direction('top')
    # Useless
    ax.axis['bottom'].set_visible(False)
    # Contours along standard deviations
    ax.grid()
    self._ax = ax # Graphical axes
    self.ax = ax.get_aux_axes(tr) # Polar coordinates
    # Add reference point and STD contour
    l , = self.ax.plot([0], self.STD[0], 'r*', ls='', ms=12, label=label[0])
    l1 , = self.ax.plot([0], self.STD[0], 'r*', ls='', ms=12, label=label[0])
#    q , = self.ax.plot([0], self.STD[1], 'b*', ls='', ms=12, label=label[1])
#    q1 , = self.ax.plot([0], self.STD[1], 'b*', ls='', ms=12, label=label[1])
    t = np.linspace(0, (np.pi / 2.0))
    t1 = np.linspace(0, (np.pi / 2.0))
    r = np.zeros_like(t) + self.STD[0]
    r1 = np.zeros_like(t) + self.STD[0]
#    p = np.zeros_like(t) + self.STD[1]
#    p1 = np.zeros_like(t) + self.STD[1]
    self.ax.plot(t, r, 'k--', label='_')
#    self.ax.plot(t, p, 'b--', label='_')
    # Collect sample points for latter use (e.g. legend)
    self.samplePoints = [l]
#    self.samplePoints = [l1]
    #self.samplePoints.append(q)
#    self.samplePoints.append(q1)
  def add_sample(self,STD,r,*args,**kwargs):
    l,= self.ax.plot(np.arccos(r), STD, *args, **kwargs) # (theta, radius)
    self.samplePoints.append(l)
    return l

#  def add_sample(self,STD,r1,*args,**kwargs):
#    l1,= self.ax.plot(np.arccos(r1), STD, *args, **kwargs) # (theta, radius)
#    self.samplePoints.append(l1)
#    return l1

  def add_contours(self,component,color,levels=5,**kwargs):
    rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
    RMSE=np.sqrt(np.power(self.STD[component], 2) + np.power(rs, 2) - (2.0 * self.STD[component] * rs  *np.cos(ts)))
    contours = self.ax.contour(ts, rs, RMSE, levels, colors=color)
    return contours

def srl(obsSTD, s, r, l, fname,markers,output_plot_folder_comparison):
    fig=plt.figure(figsize=(20,16))
    dia=TaylorDiagram(obsSTD, fig=fig, rect=111, label=['ref'])
    plt.clabel(dia.add_contours(0,'k'), inline=1, fontsize=40)
    #plt.clabel(dia.add_contours(1,'b'), inline=1, fontsize=20)
    srlc = zip(s, r, l)
    #srlc1 = zip(s1, r1, l1)

    for count,i in enumerate(srlc):
        dia.add_sample(i[0], i[1], label=i[2], marker=markers[count],markersize=12,mec = 'red', mfc = 'none', mew=1.6)
#  for count,i in enumerate(srlc1):
#    dia.add_sample(i[0], i[1], label=i[2], marker=markers[count],markersize=12, mec = 'blue', mfc = 'none', mew=1.6)
        spl = [p.get_label() for p in dia.samplePoints]
        fig.legend(dia.samplePoints, spl, numpoints=1, prop={'size': 20},  loc=[0.83,0.55])
        fig.suptitle("Taylor Diagram for " + fname.split('_TaylorDiagram.png')[0], fontsize=35)
        fig.savefig(output_plot_folder_comparison+'/'+fname)
