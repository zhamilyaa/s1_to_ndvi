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
# storage_folder = './'

def wkt_to_geojson(geometry):

    polygon = shapely.wkt.loads(geometry)
    new_name = 'salem'
    tiles_folder = Path(storage_folder.joinpath(str('415996db3e0605e6a93ac2e7b565ebc4')+'_preprocessed_images')).absolute()
    wkt_to_geojson = str(new_name)+'.geojson'
    g2 = shapely.geometry.mapping(polygon)
    with open(tiles_folder.joinpath(wkt_to_geojson), 'w') as dst:
        json.dump(g2, dst)
    logger.debug("cool")
    return


def do_s1_to_ndvi(geometry, zip_path):
    polygon = shapely.wkt.loads(geometry)

    new_name = hashlib.md5((str(zip_path)+str(polygon)).encode('utf-8')).hexdigest()

    path = Path(__file__).absolute().parents[1]

    vh_name = str(new_name)+'_vh.tif'
    vv_name = str(new_name)+'_vv.tif'
    nrpb_name = str(new_name)+'_nrpb.tif'
    lia_name = str(new_name)+'_lia.tif'
    cropped = str(new_name)+'_cropped_new_area.tif'
    wkt_to_geojson = str(new_name)+'.geojson'
    vrt = str(new_name)+'.vrt'
    result_image = str(new_name)+'_res.tif'

    tiles_folder = Path(storage_folder.joinpath(str(new_name)+'_preprocessed_images')).absolute()
    tiles_folder.mkdir(exist_ok=True)

    preprocess = str("/opt/snap/bin/gpt") + " " + str(path.joinpath(
        'preprocessing.xml')) + " -Pfilter='Lee' -Porigin=30 -Pdem='SRTM 1Sec HGT' -Presolution=10 -Pcrs='GEOGCS[" + '"WGS84(DD)"' + ", DATUM[" + '"WGS84"' + ", SPHEROID[" + '"WGS84"' + ", 6378137.0, 298.257223563]], PRIMEM[" + '"Greenwich"' + ", 0.0], UNIT[" + '"degree"' + ", 0.017453292519943295], AXIS[" + '"Geodetic longitude"' + ", EAST], AXIS[" + '"Geodetic latitude"' + ", NORTH]]' -Ssource=" + str(
        zip_path) + " -Poutput_vh=" + str(tiles_folder.joinpath(vh_name)) + " -Poutput_vv=" + str(
        tiles_folder.joinpath(vv_name)) + " -Poutput_nrpb=" + str(
        tiles_folder.joinpath(nrpb_name)) + " -Poutput_lia=" + str(tiles_folder.joinpath(lia_name))
    os.system(preprocess)

    merge_nodes = 'gdalbuildvrt -separate '+str(tiles_folder.joinpath(vrt))+' '+str(tiles_folder.joinpath(vh_name))+' '+str(tiles_folder.joinpath(vv_name))+' '+str(tiles_folder.joinpath(nrpb_name))+' '+str(tiles_folder.joinpath(lia_name))
    os.system(merge_nodes)

    g2 = shapely.geometry.mapping(polygon)
    with open(tiles_folder.joinpath(wkt_to_geojson), 'w') as dst:
        json.dump(g2, dst)

    crop = "gdalwarp -crop_to_cutline -cutline "+str(tiles_folder.joinpath(wkt_to_geojson))+" "+str(tiles_folder.joinpath(vrt))+" "+str(tiles_folder.joinpath(cropped))+" -t_srs EPSG:3857 -dstnodata 0"
    os.system(crop)
    return


def train():
    image = '/Users/zhamilya/Desktop/storage/data/5e4391ac5a5bf6bd8e7128ebabc7e561_preprocessed_images/training.tif'
    with rasterio.open(image) as file:
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


def test():
    image = '/Users/zhamilya/Desktop/storage/data/a66e3b2971da2cb736b3b124ab07d40c_preprocessed_images/testing_data_upd.tif'
    with rasterio.open(image) as file:
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
    filename = "randomforest_slm2_100_25.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model)

    yhat = loaded_model.predict(x)
    print(yhat.shape)

    yhat_img = yhat.reshape(1,3533, 3419)
    print(yhat_img.shape)

    with rasterio.open(image) as src:
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

def train_xgboost():
    image = '/Users/zhamilya/Desktop/storage/data/a66e3b2971da2cb736b3b124ab07d40c_preprocessed_images/training_data.tif'
    with rasterio.open(image) as file:
        channels = file.read()

    logger.info(f"INPUT TIF SHAPE {channels.shape}")
    data = np.moveaxis(channels, 0, -1)
    logger.info(f"INPUT TIF SHAPE {data.shape}")
    data = data.reshape(-1,5)
    logger.debug(data.shape)

    logger.debug(data.shape)
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=10887084, replace=False)
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

    geometry = 'Polygon ((68.59442325395232842 53.98420560633174148, 68.62193240235221481 53.76255184966767331, 69.02234334017269646 53.74447955715930192, 69.01928676812825358 53.98600285712484492, 69.01928676812825358 53.98600285712484492, 68.59442325395232842 53.98420560633174148))'
    zip_path = '/home/zhamilya/PycharmProjects/s1_to_ndvi/S1B_IW_GRDH_1SDV_20210609T014215_20210609T014240_027273_0341F1_B9D6.SAFE'
    do_s1_to_ndvi(geometry, zip_path)
    logger.debug("finished")
    return


if __name__ == '__main__':
    main()
