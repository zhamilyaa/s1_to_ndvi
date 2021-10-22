import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lgblkb_tools import logger
# from osgeo import gdal
from torch.utils.data import DataLoader, Dataset, random_split
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from pickle import dump
from pickle import load
import albumentations as A
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import cv2
import logging
from pytorch_lightning.loggers import WandbLogger  # newline 1
from pytorch_lightning import Trainer
from box import Box
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

scaler_b1 = load(open('scaler_b1.pkl', 'rb'))
scaler_b2 = load(open('scaler_b2.pkl', 'rb'))
scaler_b3 = load(open('scaler_b3.pkl', 'rb'))


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

# DATASET PREPROCESSING
class TheDataset(pl.LightningDataModule):
    def __init__(self, path, files, transform=None):
        self.path = path
        self.dirs = files
        self.transforms = transform

    # GLOBAL SCALES
    @staticmethod
    def read_tif(path='./res/new_data.tif'):
        with rasterio.open(path) as ds:
            data = ds.read()
            d = dict(s1=data[1:], s2=data[:1], path=path)
            d1 = d['s1']

            s1_b1 = d1[0, :, :].reshape(-1, 1)
            s1_b2 = d1[1, :, :].reshape(-1, 1)
            s1_b3 = d1[2, :, :].reshape(-1, 1)
            s1_b3 = np.clip(s1_b3, 0, 0.75)

            scaler_b1 = StandardScaler().fit(s1_b1)
            scaler_b2 = StandardScaler().fit(s1_b2)
            scaler_b3 = StandardScaler().fit(s1_b3)

            dump(scaler_b1, open('./scaler_b1.pkl', 'wb'))
            dump(scaler_b2, open('./scaler_b2.pkl', 'wb'))
            dump(scaler_b3, open('./scaler_b3.pkl', 'wb'))

            return

    def read_sample(self, item):
        path = os.path.join(self.path, self.dirs[item])
        with rasterio.open(path) as ds:
            data = np.moveaxis(ds.read(), 0, -1)
            return dict(sentinel=data, path=path)

    def __getitem__(self, item):
        s = self.read_sample(item)
        sentinel = s['sentinel']

        # FOR TEST DATA THERE IS NO TRANSFORM
        if self.transforms == None:
            sentinel = sentinel
        else:
            sentinel = self.transforms(image=sentinel)['image']

        s1 = sentinel[:, :, 1:]
        s1_b1 = s1[:, :, 0].reshape(-1, 1)
        s1_b2 = s1[:, :, 1].reshape(-1, 1)
        s1_b3 = s1[:, :, 2].reshape(-1, 1)

        s2 = sentinel[:, :, :1]
        s2_b = s2.reshape(-1, 1)

        # SCALING WITH GLOBAL SCALERS OF EACH BAND
        s1_b1_tensor = torch.from_numpy((scaler_b1.transform(s1_b1)).reshape(1, 128, 128))
        s1_b2_tensor = torch.from_numpy((scaler_b2.transform(s1_b2)).reshape(1, 128, 128))
        s1_b3_tensor = torch.from_numpy((scaler_b3.transform(s1_b3)).reshape(1, 128, 128))
        s2_b_tensor = torch.from_numpy(s2_b.reshape(1, 128, 128))

        # CONCATINATING BANDS
        s1 = torch.cat((s1_b1_tensor, s1_b2_tensor,s1_b3_tensor))
        s2 = s2_b_tensor

        path = s['path']

        return s1, s2, path

    def __len__(self):
        return len(self.dirs)

# GET LOADERS
class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, path, files, transform):
        self.batch_size = batch_size
        self.path = path
        self.files = files
        self.transform = transform

    def train_dataloader(self):
        train_data = TheDataset(self.path, self.files, self.transform)
        logger.info(f"Length of train data: {len(train_data)} ")
        return DataLoader(dataset=train_data,
                                  batch_size=self.batch_size,
                                  shuffle=False)

    def train_dataloader(self):
        test_data = TheDataset(self.path, self.files, self.transform)
        logger.info(f"Length of test data: {len(test_data)} ")
        return DataLoader(dataset=test_data,
                                  batch_size=self.batch_size,
                                  shuffle=False)

# MODEL
class ResUnetPlusPlus(LightningModule):
    def __init__(self, channel, filters=(6, 32, 64, 128, 256)):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        return out

    #
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)
        wandb.log({"Leaning rate": optimizer.param_groups[0]["lr"]})
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(
                        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, )
                    ))

    def training_step(self, train_batch, batch_idx):
        x, y, path = train_batch
        logits = self(x)
        loss = nn.MSELoss()(logits, y)
        sch = self.lr_schedulers()
        if (batch_idx + 1) % 4 == 0:
            sch.step()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 4 == 0:
            sch.step()
        wandb.log({"Training loss (average)": loss.detach().item()})
        images = x, y, logits
        wandb.log({f"Epoch worst": [wandb.Image(i) for i in images]})

        return loss

    def test_step(self, val_batch, batch_idx):
        x, y, path = val_batch
        logits = self(x)
        loss = nn.MSELoss(logits, y)
        wandb.log({"Test loss (average)": loss.detach().item()})
        images = x, y, logits
        wandb.log({f"Epoch worst": [wandb.Image(i) for i in images]})
        return loss

    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #     loss.backward()



def run_forest_run(dm: DataModule):
    # logger.setLevel(logging.DEBUG)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    logger.info("X_train.shape: %s", X_train.shape)
    logger.info("X_test.shape: %s", X_test.shape)
    regressor = RandomForestRegressor(verbose=True, n_jobs=-1, n_estimators=100)
    regressor.fit(X_train, y_train)
    yhat = regressor.predict(X_test)
    mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y_test))
    score = regressor.score(X_test, y_test)
    logger.info("score:\n%s", score)
    logger.info("mse_loss: %s", mse_loss)


    pass



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    logger.info('Start')
    TheDataset.read_tif()
    logger.info("GLOBAL SCALING DONE.")


    path = './tiles_256'
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(),
    ])

    wandb.init(project="project_template", entity='zhami', config={
        "learning_rate": 1e-2,
        "architecture": "ResUnetPlusPlus",
        "batch_size": 4})

    dir = os.listdir(path)
    train, test = train_test_split(dir, test_size=0.2)
    logger.info("DATA LOADER STARTS")
    train_load = DataModule(batch_size=wandb.config.batch_size, path = path, files=train, transform=transform)
    run_forest_run(train_load)


    train_dataloader = train_load.train_dataloader()
    test_load = DataModule(batch_size=wandb.config.batch_size, path = path, files=test, transform=None)
    test_dataloader = test_load.train_dataloader()
    logger.info("DATA LOADER ENDS")
    # logger.info("TRAIN ENUMERATING STARTS")
    # for b, (x_train, y_train, path) in enumerate(train_dataloader):
    #     x_train, y_train, path = x_train, y_train, path
    # logger.info("TRAIN ENUMERATING ENDS")
    # regressor = RandomForestRegressor(verbose=True, n_jobs=-1, n_estimators=500)
    # regressor.fit(x_train, y_train)
    # logger.info("TEST ENUMERATING STARTS")
    # for b, (x_test,y_test, path) in enumerate(test_dataloader):
    #     x_test, y_test, path = x_test,y_test, path
    # logger.info("TEST ENUMERATING ENDS")
    # yhat = regressor.predict(x_test)
    # logger.info("TEST ENDS")
    # mse_loss = nn.MSELoss()(torch.from_numpy(yhat), torch.from_numpy(y_test))
    # score = regressor.score(x_test, y_test)
    # logger.info("score:\n%s", score)
    # logger.info("mse_loss: %s", mse_loss)
    #
    #
    # return


    wandb_logger = WandbLogger()  # newline 2
    model = ResUnetPlusPlus(3,)
    wandb.watch(model)
    trainer = pl.Trainer(gpus=1, logger=wandb_logger)
    trainer.fit(model, train_dataloader, test_dataloader)





if __name__ == '__main__':
    main()
