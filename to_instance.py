'''Converts a binary mask ([0,0,0],[1,1,1]) to an instance segmentation mask'''

import os
import sys
import geojson
import numpy as np
from PIL import Image, ImageDraw
from osgeo import osr, ogr, gdal

def polygonize(mask):
    """
    Create a geojson of building footprints from input mask.
    Args:
        mask (str): path to mask to polygonize
    """
    # Polygonize with gdal
    src_ds = gdal.Open(mask)
    drv = ogr.GetDriverByName('GeoJSON')
    dst_ds = drv.CreateDataSource(mask[:-3] + 'geojson')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())
    dst_layer = dst_ds.CreateLayer(mask[:-4], srs=srs)
    dst_fieldname = 'DN'
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    gdal.Polygonize(src_ds.GetRasterBand(1), None, dst_layer, 0)
    src_ds, dst_ds = None, None
    return mask[:-3] + 'geojson'

def burn_buildings(polygon_file, side_dim=256):
    '''
    Creates a mask with different colors for each building
    Args:
        polygon_file (str): geojson file with coordinates of polygon
    '''
    with open(polygon_file) as f:
        data = geojson.load(f)

    # Create an image in which to burn polygons
    img = Image.new('L', (side_dim, side_dim), 0)
    feats = [i for i in data['features'] if i['properties']['DN'] == 1]
    try:
        ints = np.random.choice(np.arange(1,256), len(feats), replace=False)
    except:
        print(polygon_file)
        return True

    for ix, feat in enumerate(feats):
        vertices = [(f[0], f[1]) for f in feat['geometry']['coordinates'][0]]
        fill = int(ints[ix])
        ImageDraw.Draw(img).polygon(tuple(vertices), outline=fill, fill=fill)

    img.save(polygon_file[:-7]  + 'png')


if __name__ == '__main__':
    masks = sys.argv[1]
    imgs = [os.path.join(masks, i) for i in os.listdir(masks)]

    for im in imgs:
        geoj = polygonize(im)
        burn_buildings(geoj)
        os.remove(geoj)

