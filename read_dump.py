import numpy as np
from astropy.io import ascii
import yt
from yt.units import parsec, Msun, gram, centimeter, second, Kelvin, kiloparsec
import sklearn.cluster
from yt.data_objects.particle_filters import add_particle_filter

t = ascii.read('ZAA0400_small.ascii')

dataarr = np.array([t['x_(kpc)'], t['y_(kpc)'], t['z_(kpc)'], t['v_x_[cm/s]'],
                    t['v_y_[cm/s]'], t['v_z_[cm/s]'],]).T

_, label, _ = sklearn.cluster.k_means_.k_means(dataarr, 10)

data = {'particle_position_x': t['x_(kpc)']*kiloparsec,
        'particle_position_y': t['y_(kpc)']*kiloparsec,
        'particle_position_z': t['z_(kpc)']*kiloparsec,
        'particle_velocity_x': t['v_x_[cm/s]']*centimeter/second,
        'particle_velocity_y': t['v_y_[cm/s]']*centimeter/second,
        'particle_velocity_z': t['v_z_[cm/s]']*centimeter/second,
        'particle_mass': t['particle_mass_(g)']*gram,
        'smoothing_length': t['h_(kpc)']*kiloparsec,
        'particle_h2_abundance': t['h2_abundance'],
        'particle_co_abundance': t['co_abundance'],
        'cloud_id': label,
       }


bbox = np.array([[min(data['particle_position_x']), max(data['particle_position_x'])],
                 [min(data['particle_position_y']), max(data['particle_position_y'])],
                 [min(data['particle_position_z']), max(data['particle_position_z'])]])

loading_params = {'length_unit':kiloparsec.in_cgs(),
                  'mass_unit':gram.in_cgs(),
                  'time_unit':second.in_cgs(),
                  'velocity_unit':(centimeter/second).in_cgs(),
                 }

for ii in (1,10,100):
    d = {x: data[x][::ii] for x in data}
    ds = yt.load_particles(d,
                           periodicity=(False,False,False),
                           bbox=bbox,
                           **loading_params)


    direction = 'z'
    proj = yt.ProjectionPlot(ds, direction, ('deposit','all_density'),)
    proj.set_zlim('all', 1e-10, 1e-1)
    proj.set_cmap('all','cubehelix')
    proj.save("z_projection_every{0}th.png".format(ii))

for ii in range(4):
    d = {x: data[x][ii*(len(data[x])/4):(ii+1)*(len(data[x])/4)] for x in data}
    ds = yt.load_particles(d,
                           periodicity=(False,False,False),
                           bbox=bbox,
                           **loading_params)


    for direction in 'xyz':
        proj = yt.ProjectionPlot(ds, direction, ('deposit','all_density'),)
        proj.set_zlim('all', 1e-10, 1e-1)
        proj.set_cmap('all','cubehelix')
        proj.save("{1}_projection_{0}th_quarter.png".format(ii, direction))

def cloud_filterer(cloud_id):
    def cloud_filter(dummy, data):
        return data[('all','cloud_id') == cloud_id]
    return cloud_filter

for cloud_id in np.unique(label):
    d = {x: data[x][data['cloud_id'] == cloud_id] for x in data}
    #add_particle_filter("cloud{0}".format(cloud_id), function=cloud_filterer(cloud_id),
    #                    filtered_type='all', requires=["particle_type"])
    ds = yt.load_particles(d,
                           periodicity=(False,False,False),
                           bbox=bbox,
                           **loading_params)
    #ds.add_particle_filter("cloud{0}".format(cloud_id))
    
    #cloud_region = ds.h.all_data().cut_region(["obj['cloud_id'] == {0}".format(cloud_id)])

    for direction in 'xyz':
        proj = yt.ProjectionPlot(ds, direction, ('deposit','all_density'))#, data_source=cloud_region)
        proj.set_buff_size((1024, 1024))
        proj.set_zlim('all', 1e-10, 1e-1)
        proj.set_cmap('all','cubehelix')
        proj.save("{1}_projection_cloud{0}.png".format(cloud_id, direction))

ds = yt.load_particles(data,
                       periodicity=(False,False,False),
                       bbox=bbox,
                       **loading_params)
# currently fails proj = yt.ProjectionPlot(ds, direction, ('all','cloud_id'))
