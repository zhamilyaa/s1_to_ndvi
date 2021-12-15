import logging
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point
from pathlib import Path
import rasterio.mask
import os
import rasterio
import json
import shapely.wkt
import shapely.geometry
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from config import settings

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump, load
from sklearn.ensemble import RandomForestRegressor
from torch import nn
import torch
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
storage_folder = Path(settings.PROJECT.dirs.data_folder)


# INPUT: geometry -> in wkt, zip_path -> S1.SAFE file path
# OUTPUT: path to preprocessed cropped S1 images in tif
def do_s1_to_ndvi(geometry, zip_path):
    polygon = shapely.wkt.loads(geometry)

    new_name = hashlib.md5((str(zip_path)+str(polygon)).encode('utf-8')).hexdigest()

    path = Path(__file__).absolute().parents[1]

    # just names of the files
    vh_name = str(new_name)+'_vh.tif'
    vv_name = str(new_name)+'_vv.tif'
    nrpb_name = str(new_name)+'_nrpb.tif'
    lia_name = str(new_name)+'_lia.tif'
    cropped = str(new_name)+'_cropped_new_area.tif'
    wkt_to_geojson = str(new_name)+'.geojson'
    vrt = str(new_name)+'.vrt'

    # creating working folder
    tiles_folder = Path(storage_folder.joinpath(str(new_name)+'_preprocessed_images')).absolute()
    tiles_folder.mkdir(exist_ok=True)

    # processing S1
    preprocess = str("/opt/snap/bin/gpt") + " " + str(path.joinpath(
        'preprocessing.xml')) + " -Pfilter='Lee' -Porigin=30 -Pdem='SRTM 1Sec HGT' -Presolution=10 -Pcrs='GEOGCS[" + '"WGS84(DD)"' + ", DATUM[" + '"WGS84"' + ", SPHEROID[" + '"WGS84"' + ", 6378137.0, 298.257223563]], PRIMEM[" + '"Greenwich"' + ", 0.0], UNIT[" + '"degree"' + ", 0.017453292519943295], AXIS[" + '"Geodetic longitude"' + ", EAST], AXIS[" + '"Geodetic latitude"' + ", NORTH]]' -Ssource=" + str(
        zip_path) + " -Poutput_vh=" + str(tiles_folder.joinpath(vh_name)) + " -Poutput_vv=" + str(
        tiles_folder.joinpath(vv_name)) + " -Poutput_nrpb=" + str(
        tiles_folder.joinpath(nrpb_name)) + " -Poutput_lia=" + str(tiles_folder.joinpath(lia_name))
    os.system(preprocess)

    # creating vrt file from processed S1 images
    merge_nodes = 'gdalbuildvrt -separate '+str(tiles_folder.joinpath(vrt))+' '+str(tiles_folder.joinpath(vh_name))+' '+str(tiles_folder.joinpath(vv_name))+' '+str(tiles_folder.joinpath(nrpb_name))+' '+str(tiles_folder.joinpath(lia_name))
    os.system(merge_nodes)

    # converting wkt to geojson
    g2 = shapely.geometry.mapping(polygon)
    with open(tiles_folder.joinpath(wkt_to_geojson), 'w') as dst:
        json.dump(g2, dst)

    # cropping ROI from S1 vrt
    crop = "gdalwarp -crop_to_cutline -cutline "+str(tiles_folder.joinpath(wkt_to_geojson))+" "+str(tiles_folder.joinpath(vrt))+" "+str(tiles_folder.joinpath(cropped))+" -t_srs EPSG:3857 -dstnodata 0"
    os.system(crop)
    return str(tiles_folder.joinpath(cropped))


# INPUT: ndvi_path -> path to prepared ndvi tif, geom_path -> path to geojson, new_name -> encoded hashlib name (zip+polygon), cropped_path -> output of do_s1_to_ndvi
# OUTPUT: path to image for training which consist of 5 bands (S1 bands: vh,vv,nrpb, lia and S2 band: ndvi, respectively)
def s2_preparation(ndvi_path, geom_path, new_name, cropped_path):
    # entering the same working folder
    tiles_folder = Path(storage_folder.joinpath(str(new_name)+'_preprocessed_images')).absolute()

    # cropping ROI from NDVI
    cropped_ndvi = cropped_path.split(".")[0]+"_cropped_ndvi.tif"
    crop = "gdalwarp -crop_to_cutline -cutline "+str(tiles_folder.joinpath(geom_path))+" "+str(tiles_folder.joinpath(ndvi_path))+" "+str(tiles_folder.joinpath(cropped_ndvi))+" -t_srs EPSG:3857 -dstnodata 0"
    os.system(crop)

    # creating separate tifs for vh,vv,nrpb and lia to create further vrt file with separate bands
    vh_path = cropped_path.split(".")[0]+"_vh.tif"
    vv_path = cropped_path.split(".")[0]+"_vv.tif"
    nrpb_path = cropped_path.split(".")[0]+"_nrpb.tif"
    lia_path = cropped_path.split(".")[0]+"_lia.tif"

    vh = f'gdal_translate {cropped_path} -b {int(1)} {vh_path}'
    os.system(vh)
    vv = f'gdal_translate {cropped_path} -b {int(2)} {vv_path}'
    os.system(vv)
    nrpb = f'gdal_translate {cropped_path} -b {int(3)} {nrpb_path}'
    os.system(nrpb)
    lia = f'gdal_translate {cropped_path} -b {int(4)} {lia_path}'
    os.system(lia)

    # gdalbuildvrt separate each band, now including ndvi
    img_train_vrt_path = cropped_path.split(".")[0]+"_train.vrt"
    img_train_vrt = f'gdalbuildvrt -separate {img_train_vrt_path} {vh_path} {vv_path} {nrpb_path} {lia_path} {cropped_ndvi}'
    os.system(img_train_vrt)

    # training vrt to training tif
    img_train_path = cropped_path.split(".")[0]+"_train.tif"
    img_train = f'gdal_translate {img_train_vrt_path} {img_train_path}'
    os.system(img_train)
    return str(img_train)

def train(img_train_path):
    with rasterio.open(img_train_path) as file:
        channels = file.read()

    logger.info(f"INPUT TIF SHAPE {channels.shape}")
    data = np.moveaxis(channels, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1,5)
    logger.debug(data.shape)

    logger.debug(data.shape)
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=data.shape[0], replace=False)
    data = data[random_indices, :]

    valid_pixels = ~np.isnan(data.sum(axis=1))  # .reshape(-1, 1)
    print((valid_pixels==0).sum())
    print(valid_pixels.shape)

    logger.info(f"INPUT TIF SHAPE {data.shape}")

    s1_b1 = data[valid_pixels, 0].reshape(-1,1)
    s1_b2 = data[valid_pixels, 1].reshape(-1,1)
    s1_b3 = data[valid_pixels, 2].reshape(-1,1)
    s1_b4 = data[valid_pixels, 3].reshape(-1,1)

    scaler_b1 = StandardScaler().fit(s1_b1)
    scaler_b2 = StandardScaler().fit(s1_b2)
    scaler_b3 = StandardScaler().fit(s1_b3)
    scaler_b4 = StandardScaler().fit(s1_b4)

    dump(scaler_b1, open('./scaler_b1.pkl', 'wb'))
    dump(scaler_b2, open('./scaler_b2.pkl', 'wb'))
    dump(scaler_b3, open('./scaler_b3.pkl', 'wb'))
    dump(scaler_b4, open('./scaler_b4.pkl', 'wb'))

    s1_b1_tensor = ((scaler_b1.transform(s1_b1)).reshape(-1, 1))
    s1_b2_tensor = ((scaler_b2.transform(s1_b2)).reshape(-1, 1))
    s1_b3_tensor = ((scaler_b3.transform(s1_b3)).reshape(-1, 1))
    s1_b4_tensor = ((scaler_b4.transform(s1_b4)).reshape(-1, 1))

    s2_b_tensor = data[valid_pixels, 4].reshape(-1,1)

    s1 = np.concatenate((s1_b1_tensor, s1_b2_tensor,s1_b3_tensor, s1_b4_tensor), axis = 1)
    s2 = s2_b_tensor

    print(s1.shape, type(s1))
    print(s2.shape, type(s2))

    X_train, X_test, y_train, y_test = train_test_split(s1, s2, test_size=0.1)
    logger.info(f"TRAIN X SHAPE {X_train.shape}")
    logger.info(f"TRAIN Y SHAPE {y_train.shape}")
    logger.info(f"TEST X SHAPE {X_test.shape}")
    logger.info(f"TEST Y SHAPE {y_test.shape}")

    regressor = RandomForestRegressor(verbose=True, n_jobs=2, n_estimators=100, max_depth = 15)
    logger.info("REGRESSOR FITTING STARTS")
    regressor.fit(X_train, y_train)
    logger.info("REGRESSOR FITTING ENDS")
    # yhat = regressor.predict(X_test)
    # logger.info("REGRESSOR PREDICTING ENDS")
    # mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y_test))
    # score = regressor.score(X_test, y_test)score = regressor.score(X_test, y_test)
    # logger.info("score:\n%s", score)
    # logger.info("score:\n%s", score)
    # logger.info("mse_loss: %s", mse_loss)
    filename = "randomforest_slm2_100_25.pkl"
    pickle.dump(regressor, open(filename, 'wb'))
    # predicted_ndvis = np.zeros(data.shape[0], 1) * np.nan
    # predicted_ndvis[valid_pixels] = regressor.predict(data[valid_pixels])

    return

def test(img_test_path):
    with rasterio.open(img_test_path) as file:
        channels = file.read()

    logger.info(f"INPUT TIF SHAPE {channels.shape}")
    data = np.moveaxis(channels, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1,4)
    logger.debug(data.shape)
    x = data[:, :]
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=data.shape[0], replace=False)
    data = data[random_indices, :]

    valid_pixels = ~np.isnan(data.sum(axis=1))  # .reshape(-1, 1)
    print((valid_pixels==0).sum())
    print(valid_pixels.shape)

    logger.info(f"INPUT TIF SHAPE {data.shape}")
    s1_b1 = data[valid_pixels, 0].reshape(-1,1)
    s1_b2 = data[valid_pixels, 1].reshape(-1,1)
    s1_b3 = data[valid_pixels, 2].reshape(-1,1)
    s1_b4 = data[valid_pixels, 3].reshape(-1,1)


    scaler_b1 = StandardScaler().fit(s1_b1)
    scaler_b2 = StandardScaler().fit(s1_b2)
    scaler_b3 = StandardScaler().fit(s1_b3)
    scaler_b4 = StandardScaler().fit(s1_b4)

    dump(scaler_b1, open('./scaler_b1.pkl', 'wb'))
    dump(scaler_b2, open('./scaler_b2.pkl', 'wb'))
    dump(scaler_b3, open('./scaler_b3.pkl', 'wb'))
    dump(scaler_b4, open('./scaler_b4.pkl', 'wb'))

    scaler_b1 = load(open('scaler_b1.pkl', 'rb'))
    scaler_b2 = load(open('scaler_b2.pkl', 'rb'))
    scaler_b3 = load(open('scaler_b3.pkl', 'rb'))
    scaler_b4 = load(open('scaler_b4.pkl', 'rb'))

    s1_b1_tensor = ((scaler_b1.transform(s1_b1)).reshape(-1, 1))
    s1_b2_tensor = ((scaler_b2.transform(s1_b2)).reshape(-1, 1))
    s1_b3_tensor = ((scaler_b3.transform(s1_b3)).reshape(-1, 1))
    s1_b4_tensor = ((scaler_b4.transform(s1_b4)).reshape(-1, 1))

    s1 = np.concatenate((s1_b1_tensor, s1_b2_tensor, s1_b3_tensor, s1_b4_tensor), axis = 1)
    print(s1.shape)
    # x = s1

    print("salem")
    filename = "randomforest_slm.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model)

    yhat = loaded_model.predict(x)
    print(yhat.shape)

    yhat_img = yhat.reshape(1,channels.shape[1], channels.shape[-1])
    print(yhat_img.shape)

    with rasterio.open(img_test_path) as src:
        out_meta = src.meta
        print(out_meta)

    out_meta.update({"driver": "GTiff",
                     "count":1},
                    nodata=0)

    with rasterio.open('test6.tiff', "w", **out_meta) as dest:
        dest.write(yhat_img)

    logger.debug("FINISHED")
    # mse_loss = nn.MSELoss()(torch.from_numpy(yhat_img), torch.from_numpy(y))
    # logger.info("mse_loss: %s", mse_loss)

def train_xgboost(img_train_path):
    with rasterio.open(img_train_path) as file:
        channels = file.read()

    logger.info(f"INPUT TIF SHAPE {channels.shape}")
    data = np.moveaxis(channels, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1,5)
    logger.debug(data.shape)

    logger.debug(data.shape)
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=data.shape[0], replace=False)
    data = data[random_indices, :]

    valid_pixels = ~np.isnan(data.sum(axis=1))  # .reshape(-1, 1)
    print((valid_pixels==0).sum())
    print(valid_pixels.shape)

    logger.info(f"INPUT TIF SHAPE {data.shape}")

    s1_b1 = data[valid_pixels, 0].reshape(-1,1)
    s1_b2 = data[valid_pixels, 1].reshape(-1,1)
    s1_b3 = data[valid_pixels, 2].reshape(-1,1)
    s1_b4 = data[valid_pixels, 3].reshape(-1,1)

    scaler_b1 = StandardScaler().fit(s1_b1)
    scaler_b2 = StandardScaler().fit(s1_b2)
    scaler_b3 = StandardScaler().fit(s1_b3)
    scaler_b4 = StandardScaler().fit(s1_b4)

    dump(scaler_b1, open('./scaler_b1.pkl', 'wb'))
    dump(scaler_b2, open('./scaler_b2.pkl', 'wb'))
    dump(scaler_b3, open('./scaler_b3.pkl', 'wb'))
    dump(scaler_b4, open('./scaler_b4.pkl', 'wb'))

    s1_b1_tensor = ((scaler_b1.transform(s1_b1)).reshape(-1, 1))
    s1_b2_tensor = ((scaler_b2.transform(s1_b2)).reshape(-1, 1))
    s1_b3_tensor = ((scaler_b3.transform(s1_b3)).reshape(-1, 1))
    s1_b4_tensor = ((scaler_b4.transform(s1_b4)).reshape(-1, 1))

    s2_b_tensor = data[valid_pixels, 4].reshape(-1,1)

    s1 = np.concatenate((s1_b1_tensor, s1_b2_tensor,s1_b3_tensor, s1_b4_tensor), axis = 1)
    s2 = s2_b_tensor

    print(s1.shape, type(s1))
    print(s2.shape, type(s2))

    X_train, X_test, y_train, y_test = train_test_split(s1, s2, test_size=0.1)
    logger.info(f"TRAIN X SHAPE {X_train.shape}")
    logger.info(f"TRAIN Y SHAPE {y_train.shape}")
    logger.info(f"TEST X SHAPE {X_test.shape}")
    logger.info(f"TEST Y SHAPE {y_test.shape}")

    regressor = XGBRegressor(n_jobs=2, n_estimators=100, max_depth = 20)
    logger.info("REGRESSOR FITTING STARTS")
    regressor.fit(X_train, y_train)
    logger.info("REGRESSOR FITTING ENDS")
    # yhat = regressor.predict(X_test)
    # logger.info("REGRESSOR PREDICTING ENDS")
    # mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y_test))
    # score = regressor.score(X_test, y_test)
    # logger.info("score:\n%s", score)
    # logger.info("mse_loss: %s", mse_loss)
    filename = "xgboost_slm_100_25.pkl"
    pickle.dump(regressor, open(filename, 'wb'))
    # predicted_ndvis = np.zeros(data.shape[0], 1) * np.nan
    # predicted_ndvis[valid_pixels] = regressor.predict(data[valid_pixels])


def main():

    logger.debug("salem")
    img_train_path = '/Users/zhamilya/Desktop/storage/data/ee73d2748fb6e51fb0f6b30f1f5f9eed_preprocessed_images/ee73d2748fb6e51fb0f6b30f1f5f9eed_cropped_new_area_train.tif'
    train(img_train_path)
    return
    geom = 'Polygon ((67.52635263107102048 54.09302907069736222, 67.52635263107102048 54.09302907069736222, 67.56743967260906913 53.20844923523117131, 67.56743967260906913 53.20844923523117131, 69.10699528788765633 53.20361546563846389, 69.10699528788765633 53.20361546563846389, 69.09732774870224148 54.0906121859010085, 69.09732774870224148 54.0906121859010085, 67.52635263107102048 54.09302907069736222))'

    zip_path = '/home/zhamilya/PycharmProjects/s1_to_ndvi/S1B_IW_GRDH_1SDV_20210609T014215_20210609T014240_027273_0341F1_B9D6.SAFE'

    polygon = shapely.wkt.loads(geom)

    new_name = hashlib.md5((str(zip_path)+str(polygon)).encode('utf-8')).hexdigest()
    cropped = str(new_name)+'_cropped_new_area.tif'
    salem = cropped.split(".")[0]+"_vh.tif"
    print(salem)

    ndvi_path = '/Users/zhamilya/Desktop/storage/data/ee73d2748fb6e51fb0f6b30f1f5f9eed_preprocessed_images/ndvi.tif'
    geom_path = '/Users/zhamilya/Desktop/storage/data/ee73d2748fb6e51fb0f6b30f1f5f9eed_preprocessed_images/ee73d2748fb6e51fb0f6b30f1f5f9eed.geojson'
    cropped = '/Users/zhamilya/Desktop/storage/data/ee73d2748fb6e51fb0f6b30f1f5f9eed_preprocessed_images/ee73d2748fb6e51fb0f6b30f1f5f9eed_cropped_new_area.tif'
    s2_preparation(ndvi_path, geom_path, new_name, cropped)
    return
    geometry = 'Polygon ((68.59442325395232842 53.98420560633174148, 68.62193240235221481 53.76255184966767331, 69.02234334017269646 53.74447955715930192, 69.01928676812825358 53.98600285712484492, 69.01928676812825358 53.98600285712484492, 68.59442325395232842 53.98420560633174148))'
    geom = 'Polygon ((67.52635263107102048 54.09302907069736222, 67.52635263107102048 54.09302907069736222, 67.56743967260906913 53.20844923523117131, 67.56743967260906913 53.20844923523117131, 69.10699528788765633 53.20361546563846389, 69.10699528788765633 53.20361546563846389, 69.09732774870224148 54.0906121859010085, 69.09732774870224148 54.0906121859010085, 67.52635263107102048 54.09302907069736222))'

    zip_path = '/home/zhamilya/PycharmProjects/s1_to_ndvi/S1B_IW_GRDH_1SDV_20210609T014215_20210609T014240_027273_0341F1_B9D6.SAFE'
    do_s1_to_ndvi(geom, zip_path)
    logger.debug("finished")
    return


if __name__ == '__main__':
    main()
