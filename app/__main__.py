from flask import Flask, render_template, render_template_string, request
from pathlib import Path
from shapely.geometry import box, GeometryCollection, MultiPolygon, Point, Polygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch import nn

import fiona
import geopandas
import hashlib
import json
import logging
import numpy as np
import os
import pickle
import random
import rasterio
import rasterio.mask
import shapely.wkt
import shapely.geometry
import torch

from config import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)

storage_folder = Path(settings.PROJECT.dirs.data_folder)


@app.route('/')
def form():
    return render_template('form.html')


def do_s1_to_ndvi(geometry, zip_path):
    polygon = shapely.wkt.loads(geometry)
    new_name = hashlib.md5((str(zip_path) + str(polygon)).encode('utf-8')).hexdigest()
    path = Path(__file__).absolute().parents[1]
    vh_name = str(new_name) + '_vh.tif'
    vv_name = str(new_name) + '_vv.tif'
    nrpb_name = str(new_name) + '_nrpb.tif'
    lia_name = str(new_name) + '_lia.tif'
    cropped = str(new_name) + '_cropped.tif'
    wkt_to_geojson = str(new_name) + '.geojson'
    vrt = str(new_name) + '.vrt'
    result_image = str(new_name) + '_res.tif'

    tiles_folder = Path(storage_folder.joinpath(str(new_name) + '_preprocessed_images')).absolute()
    tiles_folder.mkdir(exist_ok=True)

    preprocess = str("/opt/snap/bin/gpt") + " " + str(path.joinpath(
        'preprocessing.xml')) + " -Pfilter='Lee' -Porigin=10 -Pdem='SRTM 1Sec HGT' -Presolution=30 -Pcrs='GEOGCS[" \
                 + '"WGS84(DD)"' + ", DATUM[" + '"WGS84"' + ", SPHEROID[" + '"WGS84"' + \
                 ", 6378137.0, 298.257223563]], PRIMEM[" + '"Greenwich"' + ", 0.0], UNIT[" + '"degree"' + \
                 ", 0.017453292519943295], AXIS[" + '"Geodetic longitude"' + ", EAST], AXIS[" + '"Geodetic latitude"' + \
                 ", NORTH]]' -Ssource=" + str(zip_path) + " -Poutput_vh=" + str(tiles_folder.joinpath(vh_name)) + \
                 " -Poutput_vv=" + str(tiles_folder.joinpath(vv_name)) + " -Poutput_nrpb=" + \
                 str(tiles_folder.joinpath(nrpb_name)) + " -Poutput_lia=" + str(tiles_folder.joinpath(lia_name))
    os.system(preprocess)

    merge_nodes = 'gdalbuildvrt -separate ' + str(tiles_folder.joinpath(vrt)) + ' ' + \
                  str(tiles_folder.joinpath(vh_name)) + ' '+str(tiles_folder.joinpath(vv_name)) + ' ' + \
                  str(tiles_folder.joinpath(nrpb_name)) + ' ' + str(tiles_folder.joinpath(lia_name))
    os.system(merge_nodes)

    g2 = shapely.geometry.mapping(polygon)
    
    with open(tiles_folder.joinpath(wkt_to_geojson), 'w') as dst:
        json.dump(g2, dst)

    crop = "gdalwarp -crop_to_cutline -cutline " + str(tiles_folder.joinpath(wkt_to_geojson)) + " " + \
           str(tiles_folder.joinpath(vrt)) + " " + str(tiles_folder.joinpath(cropped))
    os.system(crop)

    with rasterio.open(tiles_folder.joinpath(cropped)) as src:
        out_meta = src.meta
        img = src.read()
        out_meta.update(nodata=0)
        out_meta['crs'] = "EPSG:3857"
        
        with rasterio.open(storage_folder.joinpath(cropped), 'w', **out_meta) as dst:
            dst.write(img)

    logger.info(f"INPUT TIF SHAPE {img.shape}")
    data = np.moveaxis(img, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1, 4)
    logger.debug(data.shape)
    x = data[:, 1:]
    y = data[:, 0]
    logger.info(f" X SHAPE {x.shape}")
    logger.info(f" Y SHAPE {y.shape}")

    filename = "randomforest_20.pkl"
    loaded_model = pickle.load(open(path.joinpath(filename), 'rb'))

    yhat = loaded_model.predict(x)
    yhat_img = yhat.reshape(1, 128, 128)
    print(yhat_img.shape)

    out_meta.update({
        "driver": "GTiff",
        "count": 1
    }, nodata=0)

    with rasterio.open(storage_folder.joinpath(result_image), "w", **out_meta) as dest:
        dest.write(yhat_img)

    logger.debug("FINISHED")
    mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y))
    logger.info("mse_loss: %s", mse_loss)

    return storage_folder.joinpath(result_image)


@app.route('/app/v1/s1_to_ndvi', methods=['POST'])
def perform_s1_to_ndvi():
    form_data = request.json
    return do_s1_to_ndvi(**form_data)


def main():
    # # TESTING
    # image_test = storage_folder.joinpath('files/area_check.tiff')
    # # image_test = './tile_0_64.tiff'
    # print(image_test)
    #
    # with rasterio.open(image_test) as file:
    #     channels = file.read()
    #
    # logger.info(f"INPUT TIF SHAPE {channels.shape}")
    # data = np.moveaxis(channels, 0, -1)
    # logger.info(f"INPUT TIF SHAPE {data.shape}")
    # data = data.reshape(-1,4)
    # logger.debug(data.shape)
    # x = data[:, :]
    # logger.info(f" X SHAPE {x.shape}")
    #
    # filename = "randomforest_20.pkl"
    #
    # loaded_model = pickle.load(open(filename, 'rb'))
    # print(loaded_model)
    # yhat = loaded_model.predict(x)
    # yhat_img = yhat.reshape(1,1379,2902)
    # print(yhat_img.shape)
    #
    # with rasterio.open(image_test) as src:
    #     out_meta = src.meta
    #     print(out_meta)
    #
    # out_meta.update({"driver": "GTiff",
    #                  "count":1},
    #                 nodata=0)
    #
    # with rasterio.open('final2.tiff', "w", **out_meta) as dest:
    #     dest.write(yhat_img)
    #
    # logger.debug("FINISHED")
    # # mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y))
    # # logger.info("mse_loss: %s", mse_loss)
    #
    # return
    # path = Path(__file__).absolute().parents[1]
    # filename = "randomforest_20.pkl"
    # loaded_model = pickle.load(open(path.joinpath(filename), 'rb'))
    # return
    # polygon = 'MultiPolygon (((67.68462489838638874 53.80194293541246253, 67.92794889903467492 53.87629193561054564, 67.96850289914272025 53.68366043509732322, 67.72743189850044132 53.62395593493825174, 67.68462489838638874 53.80194293541246253)))'
    # zip_path = "/home/zhamilya/PycharmProjects/s1_to_ndvi/S1B_IW_GRDH_1SDV_20210609T014215_20210609T014240_027273_0341F1_B9D6.SAFE.zip"
    #
    # new_name = '5ed24b0022afb2e5e12a9f6b3ea7cf0b'
    #
    # tiles_folder = Path(storage_folder.joinpath(str(new_name)+'_preprocessed_images')).absolute()
    # cropped = str(new_name)+'_cropped.tif'
    #
    # with rasterio.open(tiles_folder.joinpath(cropped)) as src:
    #     out_meta = src.meta
    #     img = src.read()
    #     out_meta.update(nodata=0)
    #     out_meta['crs'] = "EPSG:3857"
    #     with rasterio.open(storage_folder.joinpath(cropped), 'w', **out_meta) as dst:
    #         dst.write(img)
    #
    # logger.info(f"INPUT TIF SHAPE {img.shape}")
    # data = np.moveaxis(img, 0, -1)
    # logger.info(f"INPUT TIF SHAPE {data.shape}")
    # data = data.reshape(-1,4)
    # logger.debug(data.shape)
    # x = data[:, 1:]
    # y = data[:, 0]
    # logger.info(f" X SHAPE {x.shape}")
    # logger.info(f" Y SHAPE {y.shape}")
    # return
    geom = 'Polygon ((67.5789586670413911 54.09040740130313907, 67.59028058351294987 53.83880925749072333, 67.99283761361282075 53.84132523892885303, 67.94629195700751723 54.08285945698876418, 67.94629195700751723 54.08285945698876418, 67.5789586670413911 54.09040740130313907))'

    geometry = 'MultiPolygon (((67.68462489838638874 53.80194293541246253, 67.92794889903467492 53.87629193561054564, 67.96850289914272025 53.68366043509732322, 67.72743189850044132 53.62395593493825174, 67.68462489838638874 53.80194293541246253)))'
    zip_path = "/home/zhamilya/PycharmProjects/s1_to_ndvi/S1B_IW_GRDH_1SDV_20210609T014215_20210609T014240_027273_0341F1_B9D6.SAFE.zip"
    do_s1_to_ndvi(geom,zip_path)
    print("salem")
    return
    # app.run(host='0.0.0.0', port=5000)
    #
    # path = Path(__file__).absolute().parents[1]
    # print(path)
    #
    # zipfile = 'hello'
    # new_name = hashlib.md5(str(zipfile).encode('utf-8')).hexdigest()
    # vh_name = str(new_name) + '_vh.tif'
    # vv_name = str(new_name) + '_vv.tif'
    # nrpb_name = str(new_name) + '_nrpb.tif'
    # cropped = str(new_name) + '_cropped.tif'
    # wkt_to_geojson = str(new_name) + '.geojson'
    # vrt = str(new_name) + '.vrt'
    # new_folder = 'original'
    # processed_images = 'or'
    # preprocess = str(path.joinpath("bin/gpt")) + " " + str(path.joinpath(
    #     'preprocessing.xml')) + " -Pfilter='Lee' -Porigin=10 -Pdem='SRTM 1Sec HGT' -Presolution=10 -Pcrs='GEOGCS[" + '"WGS84(DD)"' + ", DATUM[" + '"WGS84"' + ", SPHEROID[" + '"WGS84"' + ", 6378137.0, 298.257223563]], PRIMEM[" + '"Greenwich"' + ", 0.0], UNIT[" + '"degree"' + ", 0.017453292519943295], AXIS[" + '"Geodetic longitude"' + ", EAST], AXIS[" + '"Geodetic latitude"' + ", NORTH]]' -Ssource=" + str(
    #     zipfile) + " -Poutput_vh=" + str(path.joinpath(processed_images).joinpath(vh_name)) + " -Poutput_vv=" + str(
    #     path.joinpath(processed_images).joinpath(vv_name)) + " -Poutput_nrpb=" + str(
    #     path.joinpath(processed_images).joinpath(nrpb_name))
    #
    #
    # print(preprocess)
    # return
    # with rasterio.open("./tiles_256/tile_64_64.tiff") as file:
    #     channels = file.read()
    #
    # logger.info(f"INPUT TIF SHAPE {channels.shape}")
    # data = np.moveaxis(channels, 0, -1)
    # logger.info(f"INPUT TIF SHAPE {data.shape}")
    # data = data.reshape(-1,4)
    # logger.debug(data.shape)
    # x = data[:, 1:]
    # y = data[:, 0]
    # logger.info(f" X SHAPE {x.shape}")
    # logger.info(f" Y SHAPE {y.shape}")
    # filename = "randomforest_20.pkl"
    #
    # loaded_model = pickle.load(open(filename, 'rb'))
    # print(loaded_model)
    # result = loaded_model.score(x,y)
    # logger.info("score:\n%s", result)
    # yhat = loaded_model.predict(x)
    # yhat_img = yhat.reshape(1,128,128)
    # print(yhat_img.shape)
    #
    # with rasterio.open("./tiles_256/tile_64_64.tiff") as src:
    #     out_meta = src.meta
    #     print(out_meta)
    #
    # out_meta.update({"driver": "GTiff",
    #                  "count":1},
    #                 nodata=0)
    #
    # with rasterio.open('tile4.tiff', "w", **out_meta) as dest:
    #     dest.write(yhat_img)
    #
    # logger.debug("FINISHED")
    # mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y))
    # logger.info("mse_loss: %s", mse_loss)
    #
    # return
    with rasterio.open("/Users/zhamilya/Desktop/storage/data/files/area_final.tiff") as file:
        channels = file.read()

    logger.info(f"INPUT TIF SHAPE {channels.shape}")
    data = np.moveaxis(channels, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1,5)
    logger.debug(data.shape)
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=3511656, replace=False)
    data = data[random_indices, :]

    valid_pixels = ~np.isnan(data.sum(axis=1))  # .reshape(-1, 1)
    print((valid_pixels==0).sum())
    print(valid_pixels.shape)

    # return
    # data = data[:100000, ]
    # valid_pixels = valid_pixels[:100000, ]
    logger.info(f"INPUT TIF SHAPE {data.shape}")

    X_train, X_test, y_train, y_test = train_test_split(data[valid_pixels, :3], data[valid_pixels, 4], test_size=0.1)
    logger.info(f"TRAIN X SHAPE {X_train.shape}")
    logger.info(f"TRAIN Y SHAPE {y_train.shape}")
    logger.info(f"TEST X SHAPE {X_test.shape}")
    logger.info(f"TEST Y SHAPE {y_test.shape}")

    filename = "randomforest_10.pkl"
    logger.info(filename)

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model)
    result = loaded_model.predict(X_test)
    print(len(result))

    return
    # result = loaded_model.score(X_test, y_test)
    # print(result)
    # return

    regressor = RandomForestRegressor(verbose=True, n_jobs=2, n_estimators=100, max_depth = 10)
    logger.info("REGRESSOR FITTING STARTS")
    regressor.fit(X_train, y_train)
    logger.info("REGRESSOR FITTING ENDS")
    yhat = regressor.predict(X_test)
    logger.info("REGRESSOR PREDICTING ENDS")
    mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y_test))
    score = regressor.score(X_test, y_test)
    logger.info("score:\n%s", score)
    logger.info("mse_loss: %s", mse_loss)
    filename = "randomforest_new.pkl"
    pickle.dump(regressor, open(filename, 'wb'))
    # predicted_ndvis = np.zeros(data.shape[0], 1) * np.nan
    # predicted_ndvis[valid_pixels] = regressor.predict(data[valid_pixels])

    pass
