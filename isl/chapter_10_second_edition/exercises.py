import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from ISLP.torch import ErrorTracker
from ISLP.torch.imdb import load_lookup, load_tensor, load_sparse, load_sequential
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score

from pytorch_lightning import Trainer, LightningDataModule, LightningModule
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import RMSprop
from torch import nn

from torchmetrics import Accuracy, R2Score
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize, Normalize, CenterCrop, ToTensor
import pandas as pd


def cost_derivative(beta):
    return np.cos(beta) + 1 / 10

def cost(beta):
    return np.sin(beta) + beta / 10

def gradient_descent(beta_init):
    beta_old = beta_init
    cost_old = cost(beta_old)
    eps = np.Inf

    betas = [beta_old]
    costs = [cost_old]

    while eps > 0.00001:
        beta_new = beta_old - 0.1 * cost_derivative(beta_old)
        cost_new = cost(beta_new)

        betas.append(beta_new)
        costs.append(cost_new)

        eps = abs(cost_new - cost_old)
        cost_old = cost_new
        beta_old = beta_new

    return beta_new

print(gradient_descent(2.3))
print(gradient_descent(1.4))


fig, ax = plt.subplots()
x = np.linspace(-6, 6, num=1000)
y = np.sin(x) + x / 10
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")


df = load_data("default")
y = df.pop("default")
y = np.where(y == "Yes", 1, 0)


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)


one_hot_encoder = ColumnTransformer([("one_hot_encoder", OneHotEncoder(drop="first", sparse_output=False) , ["student"])], remainder="passthrough")
scaler = StandardScaler()

cross_validated_pipeline =  Pipeline(
    [
        ("one_hot_encoder", one_hot_encoder),
        ("scaler", scaler),
        ("logistic_regression", LogisticRegressionCV(cv = KFold(5), scoring="roc_auc"))
    ]
)

cross_validated_pipeline.fit(X_train, y_train)

preds = cross_validated_pipeline.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, preds)


class DefaultDataModule(LightningDataModule):

    def __init__(self, train_dataset, validation_dataset, test_dataset,
                 batch_size=32, num_workers=4, persistent_workers=True,
                 seed=0):

        super(DefaultDataModule, self).__init__()


        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=len(self.test_dataset),
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)


class DefaultModule(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self, input_size):

        super(DefaultModule, self).__init__()


        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 1)
        )

        self.loss = nn.BCEWithLogitsLoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.01)
        self.metrics = {"accuracy": Accuracy("binary")}
        self.pre_process_y_for_metrics=lambda y: y.int()
        self.on_epoch = True

    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer


X_train_with_dummies = one_hot_encoder.fit_transform(X_train)
X_train_with_dummies = scaler.fit_transform(X_train_with_dummies)
X_test_with_dummies = one_hot_encoder.transform(X_test)
X_test_with_dummies = scaler.transform(X_test_with_dummies)

X_train_t = torch.tensor(X_train_with_dummies.astype(np.float32))
X_test_t = torch.tensor(X_test_with_dummies.astype(np.float32))
y_train_t = torch.tensor(y_train.astype(np.float32))
y_test_t = torch.tensor(y_test.astype(np.float32))

tensor_train = TensorDataset(X_train_t, y_train_t)
tensor_test = TensorDataset(X_test_t, y_test_t)

default_data_module = DefaultDataModule(tensor_train, tensor_test, tensor_test)
default_module = DefaultModule(X_train_t.shape[1])

default_trainer = Trainer(
    deterministic=True, max_epochs=40,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
default_trainer.fit(default_module, datamodule=default_data_module)
default_trainer.test(default_module, datamodule=default_data_module)

default_module.eval()

preds = default_module(X_test_t).detach().numpy()

roc_auc_score(y_test, preds)


nyse = load_data("NYSE")

for col in ["DJ_return", "log_volume", "log_volatility"]:
    for lag in range(1, 6):
        nyse[col + "_lag_" + str(lag)] = nyse[col].shift(lag)

nyse = nyse.dropna(axis=0, how="any").\
    drop(columns=["DJ_return", "log_volatility"])

nyse_no_day_of_week = nyse.drop(columns=["day_of_week"])

df_train = nyse_no_day_of_week[nyse_no_day_of_week.train].drop(columns=["train"])
df_test = nyse_no_day_of_week[~nyse_no_day_of_week.train].drop(columns=["train"])

y_train = df_train.pop("log_volume")
y_test = df_test.pop("log_volume")

linear_model = LinearRegression()
linear_model.fit(df_train, y_train)


linear_model.score(df_test, y_test)


nyse["month"] = pd.to_datetime(nyse.index).month
df_train = nyse[nyse.train].drop(columns=["train", "log_volume"])
df_test = nyse[~nyse.train].drop(columns=["train", "log_volume"])

one_hot_encoder = OneHotEncoder(drop="first")
column_transformer = ColumnTransformer(
    [("one_hot_encoder", one_hot_encoder, ["day_of_week", "month"])],
    remainder="passthrough"
)

pipeline = Pipeline(
    [("one_hot_encode", column_transformer), ("model", LinearRegression())]
)

pipeline.fit(df_train, y_train)
pipeline.score(df_test, y_test)


class NYSEDataModule(LightningDataModule):

    def __init__(self, train_dataset, validation_dataset, test_dataset,
                 batch_size=32, num_workers=4, persistent_workers=True,
                 seed=0):

        super(NYSEDataModule, self).__init__()


        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=len(self.test_dataset),
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)


class NYSEModule(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self, input_size):

        super(NYSEModule, self).__init__()


        # self.flatten = nn.Flatten()
        # self.sequential = nn.Sequential(
        #     nn.Linear(input_size, 1)
        # )
        self.dense = nn.Linear(input_size, 1)

        self.loss = nn.MSELoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.001)
        self.metrics = {"R2": R2Score()}
        self.pre_process_y_for_metrics=lambda y: y
        self.on_epoch = True

    def forward(self, x):
        # x = self.flatten(x)
        # return torch.flatten(self.sequential(x))
        return torch.flatten(self.dense(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer


one_hot_encoder = OneHotEncoder(drop="first")
column_transformer = ColumnTransformer(
    [("one_hot_encoder", one_hot_encoder, ["day_of_week", "month"])],
    remainder="passthrough"
)

pipeline = Pipeline(
    [("one_hot_encode", column_transformer), ("scaler", StandardScaler())]
)

X_train_with_dummies = pipeline.fit_transform(df_train)
X_test_with_dummies = pipeline.transform(df_test)

X_train_t = torch.tensor(X_train_with_dummies.astype(np.float32))
X_test_t = torch.tensor(X_test_with_dummies.astype(np.float32))
y_train_t = torch.tensor(y_train.astype(np.float32))
y_test_t = torch.tensor(y_test.astype(np.float32))

tensor_train = TensorDataset(X_train_t, y_train_t)
tensor_test = TensorDataset(X_test_t, y_test_t)

nyse_data_module = NYSEDataModule(tensor_train, tensor_test, tensor_test)
nyse_module = NYSEModule(X_train_t.shape[1])

nyse_trainer = Trainer(
    deterministic=True, max_epochs=40,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
nyse_trainer.fit(nyse_module, datamodule=nyse_data_module)
nyse_trainer.test(nyse_module, datamodule=nyse_data_module)

preds = nyse_module(X_test_t).detach().numpy()

r2_score(y_test, preds)


# About the same R squared score


class NYSEModuleNonLinear(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self, input_size):

        super(NYSEModuleNonLinear, self).__init__()


        # self.flatten = nn.Flatten()
        # self.sequential = nn.Sequential(
        #     nn.Linear(input_size, 1)
        # )
        self.dense = nn.Linear(input_size, 10)
        self.activation = nn.ReLU()
        self.output = nn.Linear(10, 1)

        self.loss = nn.MSELoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.001)
        self.metrics = {"R2": R2Score()}
        self.pre_process_y_for_metrics=lambda y: y
        self.on_epoch = True

    def forward(self, x):
        # x = self.flatten(x)
        # return torch.flatten(self.sequential(x))
        return torch.flatten(self.output(self.activation(self.dense(x))))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer


nyse_module_non_linear = NYSEModuleNonLinear(X_train_t.shape[1])

nyse_trainer = Trainer(
    deterministic=True, max_epochs=40,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
nyse_trainer.fit(nyse_module_non_linear, datamodule=nyse_data_module)
nyse_trainer.test(nyse_module_non_linear, datamodule=nyse_data_module)

preds = nyse_module_non_linear(X_test_t).detach().numpy()

r2_score(y_test, preds)

# About the same


cols = ["DJ_return", "log_volume", "log_volatility"]
ordered_cols = [
    "{col}_lag_{lag}".format(col=col, lag=lag)
    for lag in range(5, 0, -1) for col in cols
]

X_rnn = nyse[ordered_cols].to_numpy().reshape(-1, 5, 3)

class NYSEModuleRNN(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self):

        super(NYSEModuleRNN, self).__init__()

        self.rnn = nn.RNN(3, 12, batch_first=True)
        self.dense = nn.Linear(12, 1)
        self.dropout = nn.Dropout(0.1)

        self.loss = nn.MSELoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.001)
        self.metrics = {"R2": R2Score()}
        self.pre_process_y_for_metrics=lambda y: y
        self.on_epoch = True

    def forward(self, x):
        val, h_n = self.rnn(x)
        val = self.dense(self.dropout(val[:, -1]))
        return torch.flatten(val)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer

rnn_module = NYSEModuleRNN()

X_train_rnn_t = torch.Tensor(X_rnn[nyse.train, :, :].astype(np.float32))
X_test_rnn_t = torch.tensor(X_rnn[~nyse.train, :, :].astype(np.float32))

y_train_rnn_t = y_train_t
y_test_rnn_t = y_test_t

nyse_rnn_train = TensorDataset(X_train_rnn_t, y_train_rnn_t)
nyse_rnn_test = TensorDataset(X_test_rnn_t, y_test_rnn_t)

rnn_data_module = NYSEDataModule(nyse_rnn_train, nyse_rnn_test, nyse_rnn_test)

nyse_nn_trainer = Trainer(
    deterministic=True, max_epochs=40,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
nyse_nn_trainer.fit(rnn_module, rnn_data_module)


nyse_nn_trainer.test(rnn_module, rnn_data_module)
rnn_module.eval()

preds = rnn_module(X_test_rnn_t).detach().numpy()

r2_score(y_test, preds)


nyse = load_data("NYSE")

for col in ["DJ_return", "log_volume", "log_volatility", "day_of_week"]:
    for lag in range(1, 6):
        nyse[col + "_lag_" + str(lag)] = nyse[col].shift(lag)

nyse = nyse.dropna(axis=0, how="any").\
    drop(columns=["DJ_return", "log_volatility", "day_of_week"])

one_hot_encoder = OneHotEncoder(sparse_output=False)
column_transformer = ColumnTransformer(
    [("one_hot_encoder", one_hot_encoder, day_of_week_cols)],
    remainder="passthrough",  verbose_feature_names_out=False
).set_output(transform="pandas")

X_rnn = column_transformer.fit_transform(nyse)


cols = ["DJ_return", "log_volume", "log_volatility", "day_of_week"]

ordered_cols = []
for lag in range(5, 0, -1):
    for col in cols:
        if col == "day_of_week":
            cols_to_append = list(X_rnn.columns[X_rnn.columns.str.match(".*day_of_week_lag_" + str(lag) + ".*")])
            ordered_cols.extend(cols_to_append)
        else:
            ordered_cols.append("{col}_lag_{lag}".format(col=col, lag=lag))

X_rnn = X_rnn[ordered_cols]

df_train = X_rnn[nyse.train]
df_test = X_rnn[~nyse.train]

scaler = StandardScaler()

df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)

df_train = df_train.reshape(-1, 5, 8)
df_test = df_test.reshape(-1, 5, 8)


class NYSEModuleRNN(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self):

        super(NYSEModuleRNN, self).__init__()

        self.rnn = nn.RNN(8, 12, batch_first=True)
        self.dense = nn.Linear(12, 1)
        self.dropout = nn.Dropout(0.1)

        self.loss = nn.MSELoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.001)
        self.metrics = {"R2": R2Score()}
        self.pre_process_y_for_metrics=lambda y: y
        self.on_epoch = True

    def forward(self, x):
        val, h_n = self.rnn(x)
        val = self.dense(self.dropout(val[:, -1]))
        return torch.flatten(val)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer


rnn_module = NYSEModuleRNN()

X_train_rnn_t = torch.Tensor(df_train.astype(np.float32))
X_test_rnn_t = torch.tensor(df_test.astype(np.float32))

y_train_rnn_t = y_train_t
y_test_rnn_t = y_test_t

nyse_rnn_train = TensorDataset(X_train_rnn_t, y_train_rnn_t)
nyse_rnn_test = TensorDataset(X_test_rnn_t, y_test_rnn_t)

rnn_data_module = NYSEDataModule(nyse_rnn_train, nyse_rnn_test, nyse_rnn_test)

nyse_nn_trainer = Trainer(
    deterministic=True, max_epochs=40,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
nyse_nn_trainer.fit(rnn_module, rnn_data_module)


nyse_nn_trainer.test(rnn_module, rnn_data_module)
rnn_module.eval()

preds = rnn_module(X_test_rnn_t).detach().numpy()

r2_score(y_test, preds)

imdb_train, imdb_test = load_tensor(root='data/IMDB')


class IMDBDataModule(LightningDataModule):

    def __init__(self, train_dataset, validation_dataset, test_dataset,
                 batch_size=32, num_workers=4, persistent_workers=True,
                 seed=0):

        super(IMDBDataModule, self).__init__()


        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=len(self.test_dataset),
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

class IMDBModule(LightningModule):

    """
    A simple `pytorch_lightning` module for regression problems.
    """

    def __init__(self, input_size):

        super(IMDBModule, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Linear(64, 1)
        )

        self.loss = nn.BCEWithLogitsLoss()
        self._optimizer = RMSprop(self.parameters(), lr=0.01)
        self.metrics = {"accuracy": Accuracy("binary")}
        self.pre_process_y_for_metrics=lambda y: y.int()
        self.on_epoch = True

    def forward(self, x):
        return torch.flatten(self.sequential(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log("train_loss",
                 loss,
                 on_epoch=self.on_epoch,
                 on_step=False)

        y_ = self.pre_process_y_for_metrics(y)
        for _metric in self.metrics.keys():
            self.log(f"train_{_metric}",
                     self.metrics[_metric](preds, y_),
                     on_epoch=self.on_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

    def validation_step(self, batch, batch_idx):
        x, y = batch

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return y, self.forward(x)

    def configure_optimizers(self):
        return self._optimizer


imdb_module = IMDBModule(imdb_train.tensors[0].shape[-1])
imdb_data_module = IMDBDataModule(imdb_train, imdb_test, imdb_test)


imdb_nn_trainer = Trainer(
    deterministic=True, max_epochs=10,
    log_every_n_steps=5, callbacks=[ErrorTracker()]
)
imdb_nn_trainer.fit(imdb_module, imdb_data_module)

imdb_nn_trainer.test(imdb_module, imdb_data_module)

resize = Resize((232,232))
crop = CenterCrop (224)
normalize = Normalize([0.485,0.456,0.406], [0.229 ,0.224 ,0.225])
imgfiles = sorted([f for f in glob('my_images/*')])
imgs = torch.stack([torch.div(crop(resize(read_image(f))), 255) for f in imgfiles])
imgs = normalize(imgs)

resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.eval()

img_preds = resnet_model(imgs)

img_probs = np.exp(np.asarray(img_preds.detach()))
img_probs /= img_probs.sum(1)[:,None]

labs = json.load(open('imagenet_class_index.json'))
class_labels = pd.DataFrame([(int(k), v[1]) for k, v in labs.items()], columns=['idx', 'label'])
class_labels = class_labels.set_index('idx')
class_labels = class_labels.sort_index()

for i, imgfile in enumerate(imgfiles):
    img_df = class_labels.copy()
    img_df['prob'] = img_probs[i]
    img_df = img_df.sort_values(by='prob', ascending=False)[:3]
    print(f'Image: {imgfile}')
    print(img_df.reset_index().drop(columns=['idx']))
