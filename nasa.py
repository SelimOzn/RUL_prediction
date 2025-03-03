from collections import Counter
import tensorflow as tf
import kagglehub
import matplotlib
import pandas as pd
import numpy as np
import sklearn
from keras.src.backend import backend
from imblearn.over_sampling import RandomOverSampler
from keras.src.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.src.optimizers import AdamW
from matplotlib import rcParams, matplotlib_fname, style, cm
from matplotlib.colors import Normalize
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras import backend as K
from keras import Sequential, Model
from keras.layers import LSTM, Dense, InputLayer, Dropout, Activation
from tensorflow.python.keras import Input
from tensorflow.python.keras.utils.version_utils import callbacks
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)


# Download latest version
path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

print("Path to dataset files:", path)

train = pd.read_csv("{}/CMaps/train_FD001.txt".format(path), sep=" ", header=None)
test = pd.read_csv("{}/CMaps/test_FD001.txt".format(path), sep=" ", header=None)

print(train.describe())
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

train.drop(columns=[26,27], inplace=True)
test.drop(columns=[26,27], inplace=True)

train.columns = columns
test.columns = columns

print(train["P15"].describe())
train.drop(columns=["TRA", "T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd"], inplace=True)

def prepare_train_data(data, factor=0):
    df = data.copy()
    a = df.groupby("unit_number")
    b = a["time_in_cycles"]
    c = b.max()
    fd_RUL = c.reset_index()
    fd_RUL.columns = ['unit_number', 'max']
    df = df.merge(fd_RUL, how="left")
    check = df.merge(fd_RUL, how="right")
    print(df.equals(check))
    df["RUL"] = df["max"] - df["time_in_cycles"]
    df.drop(columns = ["max"], inplace=True)
    return df[df["time_in_cycles"] > factor]

df = prepare_train_data(train)
corr = df.corr()
print(plt.get_fignums())
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, cmap="RdYlGn", linewidths=0.2)

# print(plt.get_fignums())
# fig=plt.gcf()
# print(plt.get_fignums())
# fig.set_size_inches(20,20)

plt.show()

def score(y_true, y_pred, a1=10, a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0:
            score += math.exp(i/a2) - 1
        else:
            score += math.exp(-i/a1) - 1
    return score

def score_func(y_true, y_pred):
    lst = [round(score(y_true, y_pred), 2),
           round(mean_absolute_error(y_true, y_pred), 2),
           round(mean_squared_error(y_true, y_pred), 2)**0.5,
           round(r2_score(y_true, y_pred), 2)]

    print(f"Compatitive Score {lst[0]}")
    print(f"Mean Absolute Error {lst[1]}")
    print(f"Mean Squared Error {lst[2]}")
    print(f"R2 Score {lst[3]}")
    return [lst[1], round(lst[2],2), lst[3]*100]

unit_number = pd.DataFrame(df["unit_number"])
train_df_1 = df.drop(columns=["unit_number", "setting_1", "setting_2", "P15", "NRc"])
print(train_df_1.head())

def lstm_data_preprocessing(raw_train_data, raw_test_data, raw_RUL_data):
    train_df = raw_train_data.copy()
    truth_df = raw_RUL_data.copy()
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    w1 = 80
    w0 = 30
    train_df["cycle_norm"] = train_df["time_in_cycles"]
    train_df["label1"] = np.where(train_df["RUL"] <= w1, 1, 0)
    train_df["label2"] = train_df["label1"]
    train_df.loc[train_df["RUL"] <= w0, "label2"] = 2



    cols_normalize = train_df.columns.difference(["unit_number", "time_in_cycles", "RUL", "label1", "label2"])
    deneme = train_df.copy()
    train_min = train_df[cols_normalize].min()
    train_max = train_df[cols_normalize].max()

    min_max_scaler = MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize, index=train_df.index)

    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    print("train_df >> ", train_df.head(), "\n\n")

    test_df = raw_test_data.drop(columns = ["setting_1", "setting_2", "P15", "NRc", "max"])
    test_df["cycle_norm"] = test_df["time_in_cycles"]

    for col in cols_normalize:
        print(f"{col}: Train Mean={deneme[col].mean()}, Test Mean={test_df[col].mean()}")

    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), columns=cols_normalize, index=test_df.index)

    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)

    rul = pd.DataFrame(test_df.groupby("unit_number")["time_in_cycles"].max()).reset_index()
    rul.columns = ['unit_number', 'max']
    truth_df.columns = ["more"]
    truth_df["unit_number"] = truth_df.index + 1
    truth_df["max"] = truth_df["more"] + rul["max"]
    truth_df.drop("more", inplace=True, axis=1)

    test_df = test_df.merge(truth_df, how="left", on="unit_number")
    test_df["RUL"] = test_df["max"] - test_df["time_in_cycles"]
    test_df.drop("max", inplace=True, axis=1)

    test_df["label1"] = np.where(test_df["RUL"] <= w1, 1, 0)
    test_df["label2"] = test_df["label1"]
    test_df.loc[test_df["RUL"] <= w0, "label2"] = 2
    print("test_df >> ", test_df.head(), "\n\n")

    print("RUL: Train Mean=",deneme["RUL"].mean(), "Test Mean=", test_df["RUL"].mean())


    sequence_length = 50

    def gen_sequence(id_df, seq_length, seq_cols):
        matrix = id_df[seq_cols].values
        num_elements = matrix.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield matrix[start:stop, :]

    sequence_cols = list(test_df.columns[:-3])
    print(sequence_cols)

    val = list(gen_sequence(train_df[train_df["unit_number"] == 1], sequence_length, sequence_cols))
    print(len(val), len(val[0]), len(val[0][0])) #Deneme

    seq_gen = (np.array(list(gen_sequence(train_df[train_df["unit_number"] == id], sequence_length, sequence_cols)))
           for id in train_df["unit_number"].unique())

    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

    def gen_labels(id_df, seq_length, label):
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        return data_matrix[seq_length:num_elements, :]

    label_gen = [gen_labels(train_df[train_df["unit_number"] == id], sequence_length, ["RUL"])
                 for id in train_df["unit_number"].unique()]

    #İlk pencere 0'dan 49'a kadar iken ilk label 50.verinin RUL'u.

    label_array = np.concatenate(label_gen).astype(np.float32)
    print("Train seti shape:", seq_array.shape, label_array.shape)

    print("Train Min Values:\n", train_df[cols_normalize].min())
    print("Train Max Values:\n", train_df[cols_normalize].max())

    print("Test Min Values:\n", test_df[cols_normalize].min())
    print("Test Max Values:\n", test_df[cols_normalize].max())

    return seq_array, label_array, test_df, sequence_length, sequence_cols, train_df

def r2_keras(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lstm_train(seq_array, label_array, sequence_length):
    nb_features = seq_array.shape[2]
    nb_output = label_array.shape[1]

    model = Sequential()
    model.add(InputLayer(shape=(sequence_length, nb_features)))
    model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, return_sequences=False, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(units=nb_output, activation="linear"))

    # AdamW optimizer ve ReduceLROnPlateau kullan
    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["mae", r2_keras])

    # Öğrenme hızı zamanla düşsün
    lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    # Modeli eğit
    history = model.fit(seq_array, label_array, epochs=60, batch_size=200, validation_split=0.05, verbose=2,
                        callbacks=[lr_schedule])
    print(history.history.keys())

    return model, history

def lstm_test_evaluation_graphs(model, history, seq_array, label_array):
    # summarize history for R^2
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('model r^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_acc.savefig("model_r2.png")

    # summarize history for MAE
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_acc.savefig("model_mae.png")

    # summarize history for Loss
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_acc.savefig("model_regression_loss.png")

    # training metrics
    scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
    print('\nMAE: {}'.format(scores[1]))
    print('\nR^2: {}'.format(scores[2]))

    y_pred = model.predict(seq_array,verbose=1, batch_size=200)
    y_true = label_array

    test_set = pd.DataFrame(y_pred)
    test_set.head()
    test_set.to_csv('submit_train.csv', index=False)

def lstm_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols):
    seq_array_test_last = [lstm_test_df[lstm_test_df['unit_number']==id][sequence_cols].values[-sequence_length:]
                           for id in lstm_test_df["unit_number"].unique()
                           if len(lstm_test_df[lstm_test_df["unit_number"]==id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    y_mask = [len(lstm_test_df[lstm_test_df["unit_number"]==id])>= sequence_length
              for id in lstm_test_df["unit_number"].unique()]

    label_array_test_last = lstm_test_df.groupby("unit_number")["RUL"].nth(-1)
    label_array_test_last = label_array_test_last[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], -1).astype(np.float32)
    #OR
    #label_array_test_last = lstm_test_df.groupby("unit_number")[["RUL"]].nth(-1)[y_mask].values
    estimator = model
    print("Son window testinde shape", seq_array_test_last.shape, label_array_test_last.shape)
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2, batch_size=200)

    print('\nMAE: {}'.format(scores_test[1]))
    print('\nR^2: {}'.format(scores_test[2]))

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last

    test_set = pd.DataFrame(y_pred_test)
    print(test_set.head())

    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test, color='blue')
    plt.plot(y_true_test, color='red')
    plt.title("prediction")
    plt.xlabel("row")
    plt.ylabel("value")
    plt.legend(["predicted", "true"], loc='upper left')
    plt.show()
    fig_verify.savefig("model_regression_verify.png")

    return scores_test[1], scores_test[2]



def train_models(data, model="FOREST"):

    if model != 'LSTM':
        X = data.iloc[:,:14].to_numpy()
        Y = data.iloc[:,14:].to_numpy()
        Y = np.ravel(Y)
        print(X.shape)
        print(Y.shape)

    if model == "FOREST":
        model = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X, Y)
        return model

    elif model == 'XGB':
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                           colsample_bytree=0.5, max_depth=3,silent=True)
        model.fit(X, Y)
        return model

    elif model == "LSTM":
        seq_array, label_array, test_df, sequence_length, sequence_cols, train_df = lstm_data_preprocessing(data[0], data[1], data[2])
        model, history = lstm_train(seq_array, label_array, sequence_length)
        return model, history, test_df, seq_array, label_array, sequence_length, sequence_cols, train_df

    return

def plot_results(y_true, y_pred):
    #print(plt.style.available)
    #style.use("ggplot")

    rcParams['figure.figsize'] = 12,10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel("training samples")
    plt.ylabel("RUL")
    plt.legend(["Predicted", "True"], loc='upper right')
    plt.title("Real vs Predicted RUL")
    plt.show()
    return

#Forest modelinde eğitim için test setin ayarlanması.
test.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)
test_max = test.groupby("unit_number")["time_in_cycles"].max().reset_index()
test_max.columns = ['unit_number', 'max']
fd_001_test = test.merge(test_max, how="left", on=["unit_number"])
test_new = fd_001_test[fd_001_test["time_in_cycles"] == fd_001_test["max"]].reset_index()
test_new.drop(columns=['index','max','unit_number','setting_1','setting_2','P15','NRc'], inplace=True)

X_001_test = test_new.to_numpy()
print(X_001_test.shape)
print(fd_001_test.head())

#setting_1, setting_2, P15, NRc olmadan eğitim
model_1 = train_models(train_df_1)

#
# RF_feature_importance = model_1.feature_importances_
# feature_names = train_df_1.columns[:-1]
# sorted_idx = np.argsort(RF_feature_importance)
# RF_sorted_feature_importance = RF_feature_importance[sorted_idx]
# plt.figure(figsize=(10, 6))
# plt.barh(np.array(feature_names)[sorted_idx], RF_sorted_feature_importance, color="royalblue")
# plt.ylabel("Feature Name")
# plt.xlabel("Feature Importance")
# plt.title("RandomForest Feature Importance")
# plt.show()
#
#
# X = train_df_1.drop(columns=["RUL"])  # Bağımlı değişkeni çıkar
# vif_data = pd.DataFrame()
# vif_data["Feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif_data)
#
# X_reduced = X.drop(columns=["Nf","NRf"])  # İlk olarak Nf'yi çıkar
# new_vif_data = pd.DataFrame()
# new_vif_data["Feature"] = X_reduced.columns
# new_vif_data["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]
# print(new_vif_data)
#
# pca = PCA(n_components=5)  # İlk 5 bileşeni al
# X = X.iloc[:,1:]
# X_pca = pca.fit_transform(X)
#
# explained_variance = pca.explained_variance_ratio_
# print("Açıklanan Varyans:", explained_variance)
#
#
# feature_importance = model_1.feature_importances_
#
# # Özellik isimleri
# feature_names = train_df_1.columns[:-1]  # RUL sütunu hariç tut
#
# # Özellikleri önem sırasına göre sırala
# sorted_idx = np.argsort(feature_importance)
#
# plt.figure(figsize=(10, 6))
# plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx], color="royalblue")
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("RandomForest Feature Importance")
# plt.show()
#



y_pred = model_1.predict(X_001_test)
RUL = pd.read_csv("{}/CMaps/RUL_FD001.txt".format(path), sep=" ", header=None)
print(RUL.head())
y_true = RUL[0].to_numpy()
RF_individual_scorelst = score_func(y_true, y_pred)
plot_results(y_true, y_pred)

#LSTM modeli için eğitim. Forest için silinen unit_number sütunu geri getirildi.
train_df_lstm = pd.concat([unit_number, train_df_1], axis=1)
def augment_rul_data(df, shift_range=50):
    augmented_df = df.copy()

    for shift in range(1, shift_range + 1, 10):  # Her 10 adımda bir veri oluştur
        df_shifted = df.copy()
        df_shifted["time_in_cycles"] -= shift  # Zamanı geriye kaydır
        df_shifted["RUL"] += shift  # RUL'ü artır

        # Negatif time_in_cycles olanları sil
        df_shifted = df_shifted[df_shifted["time_in_cycles"] > 0]

        augmented_df = pd.concat([augmented_df, df_shifted])

    return augmented_df

# Train setini genişlet
train_augmented = augment_rul_data(train_df_lstm)


model, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols, train_df =\
    train_models([train_df_lstm, fd_001_test, RUL.copy()], model="LSTM")


#Modelin performansının görselleştirilmesi.
lstm_test_evaluation_graphs(model, history, seq_array, label_array)

#Her unit için test setindeki son window üzerinde modelin performansının değerlendirilmesi
MAE, R2 = lstm_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols)
LSTM_individual_scorelst = [round(MAE,2), 0, round(R2,2)*100]


train_df_lstm_3_label = train_df_lstm.copy()
train_df_lstm_3_label["RUL_label"] = pd.cut(train_df_lstm_3_label["RUL"], bins=[0,30,80,np.inf], labels=[2,1,0])
print(train_df_lstm_3_label["RUL_label"].value_counts())
print(train_df["label2"].value_counts())




def lstm_all_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols):
    def gen_sequence(df, sequence_length, sequence_cols):
        deneme = len(df)
        num_elements = df.shape[0]
        for start, stop in zip(range(0, num_elements-sequence_length+1), range(sequence_length, num_elements+1)):
            yield df.iloc[start:stop][sequence_cols].values

    """seq_array_test = [np.array(list(gen_sequence(lstm_test_df[lstm_test_df["unit_number"]==id], sequence_length, sequence_cols)))
                      for id in lstm_test_df["unit_number"].unique()
                      if len(lstm_test_df[lstm_test_df["unit_number"]==id]) >= sequence_length]"""
    seq_array_test = []
    for id in lstm_test_df["unit_number"].unique():
        if len(lstm_test_df[lstm_test_df["unit_number"]==id]) > sequence_length:
            seq_array_test.append(np.array(list(gen_sequence(lstm_test_df[lstm_test_df["unit_number"]==id], sequence_length, sequence_cols))))
    seq_array_test = np.concatenate(seq_array_test).astype(np.float32)



    def gen_label(df, sequence_length):
        return df.iloc[sequence_length-1:][["RUL"]].values #sequence_length-1 dene


    label_array_test = [gen_label(lstm_test_df[lstm_test_df["unit_number"]==id], sequence_length)
                        for id in lstm_test_df["unit_number"].unique()
                        if len(lstm_test_df[lstm_test_df["unit_number"]==id]) > sequence_length]


    label_array_test = np.concatenate(label_array_test).astype(np.float32)
    print("Tüm test setinde shape", seq_array_test.shape, label_array_test.shape)
    estimator = model
    scores_test_all = estimator.evaluate(seq_array_test, label_array_test, verbose=2, batch_size=len(label_array_test))

    print("\nMAE: {}".format(scores_test_all[1]))
    print("\nR^2: {}".format(scores_test_all[2]))

    y_pred_test = estimator.predict(seq_array_test)
    y_true_test = label_array_test

    test_set = pd.DataFrame(y_pred_test)

    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test, color='blue')
    plt.plot(y_true_test, color='red')
    plt.title("prediction")
    plt.xlabel("row")
    plt.ylabel("value")
    plt.legend(["predicted", "true"], loc='upper left')
    plt.show()
    fig_verify.savefig("model_regression_verify.png")

    # Erken ve geç cycle'lardaki tahminleri kıyasla
    early_cycles = lstm_test_df[lstm_test_df["time_in_cycles"] < 50]
    late_cycles = lstm_test_df[lstm_test_df["time_in_cycles"] > lstm_test_df["time_in_cycles"].max() - 50]

    print("Erken cycle'lardaki gerçek RUL ortalaması:", early_cycles["RUL"].mean())
    print("Geç cycle'lardaki gerçek RUL ortalaması:", late_cycles["RUL"].mean())

    y_pred_early = estimator.predict(seq_array_test[:len(early_cycles)])
    y_pred_late = estimator.predict(seq_array_test[-len(late_cycles):])

    print("Modelin erken cycle'lardaki tahmin ortalaması:", y_pred_early.mean())
    print("Modelin geç cycle'lardaki tahmin ortalaması:", y_pred_late.mean())

    return scores_test_all[1], scores_test_all[2], y_true_test, y_pred_test, seq_array_test

#Test setindeki tüm windowlar üzerinde modelin performansının değerlendirilmesi
print("Test for all")
MAE_all, R2_all, y_true_test, y_pred_test, seq_array_test = lstm_all_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols)
LSTM_all_scorelst = [round(MAE_all,2), 0, round(R2_all,2)*100]

errors = np.abs(y_true_test - y_pred_test).flatten()
cycle_numbers = (seq_array_test[:,sequence_length-1, 0:2])
rolling_errors = np.convolve(errors, np.ones(10)/10, mode="valid")

plt.figure(figsize=(12, 6))
sns.scatterplot(x=cycle_numbers[:,1], y=errors, alpha=0.5, color="red")
plt.xlabel("Time in Cycles")
plt.ylabel("Error")
plt.axhline(y=np.mean(errors), color="blue", linestyle="--", label="Mean Error")
plt.legend()
plt.show()

unit_numbers = [window[0][0] for window in seq_array_test]
unit_error_df = pd.DataFrame({"unit_number": unit_numbers,
                                    "error": errors})
unit_error_df["moving_averages"] = unit_error_df.groupby("unit_number")["error"].rolling(window=10, min_periods=1).mean().reset_index(drop=True)
unique_unit = len(unit_error_df["unit_number"].unique())
colormap = plt.get_cmap("tab20", unique_unit)

for i, unit in enumerate(unit_error_df["unit_number"].unique()):
    cycle = np.where(cycle_numbers[:, 0] == unit)[0]
    color = colormap(i)
    plt.plot(cycle, unit_error_df[unit_error_df["unit_number"]==unit]["moving_averages"].values, c=color, label=unit)

plt.xlabel("Time in Cycles")
plt.ylabel("Error")
plt.legend(ncol=2, fontsize=8, loc="best")  # Legend'ı düzenleme
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# df = prepare_train_data(train, sequence_length//2)
#
# unit_number = pd.DataFrame(df["unit_number"])
#
# train_df_1 = df.drop(columns=["unit_number", "setting_1", "setting_2", "P15", "NRc"])
# train_df_lstm = pd.concat([unit_number, train_df_1], axis=1)
# model, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols =\
#     train_models([train_df_lstm, fd_001_test, RUL.copy()], model="LSTM")
# MAE_all, R2_all = lstm_all_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols)


#Her test verisi için, test verisinin time_in_cycle'ından düşük time_in_cycle değerine sahip olan train verileri
#olmadan eğitilen modelden bir tahmin al.
def single_train(test_data, train_data, algorithm):
    y_single_pred = []
    for sample in tqdm(test_data):
        single_train_df = prepare_train_data(train_data, sample[0])
        single_train_df.drop(columns=["unit_number", "setting_1", "setting_2", "P15", "NRc"], inplace=True)
        model = train_models(single_train_df, algorithm)
        y_p = model.predict(sample.reshape(1,-1))[0]
        y_single_pred.append(y_p)
    y_single_pred = np.array(y_single_pred)
    return y_single_pred

y_single_pred = single_train(X_001_test, train, "FOREST")
plot_results(y_true, y_single_pred) #Bu grafik, önceki forest modeline göre daha iyi sonuçlar elde ettiğimizi gösteriyor.

RF_SingleTrain_scorelst = score_func(y_true, y_single_pred)

def prepare_test_data(fd_001_test, n=0):
    test = fd_001_test[fd_001_test["time_in_cycles"] == fd_001_test["max"]-n].reset_index()
    test.drop(columns=['index','max','unit_number','setting_1','setting_2','P15','NRc'], inplace=True)
    test_return = test.to_numpy()
    return test_return

#Her unit için, test setindeki son cycle'dan N önceki cycle'da olan veriler alınıyor.
#single_train bu test setine göre eğitim yapıp tahmin oluşturuyor.
#Yani single_train işlemi önce test setindeki son cycler'lar için yapıldı. Burada ise sondan bir önceki, iki önceki ...
#şeklinde N'e kadar gidecek şekilde yapıldı.
N=5
y_n_pred = y_single_pred
for i in range(1,N):
    X_001_test = prepare_test_data(fd_001_test, n=i)
    y_single_i_pred = single_train(X_001_test, train, "FOREST")
    print(y_n_pred)
    y_n_pred = np.vstack((y_n_pred, y_single_i_pred))

y_multi_pred = np.mean(y_n_pred, axis=0)
RF_5avg_scorelst = score_func(y_true, y_multi_pred)
plot_results(y_true, y_multi_pred)

N=10
y_n_pred = y_multi_pred
for i in range(5,N):
    X_001_test = prepare_test_data(fd_001_test, n=i)
    y_sing_i_pred = single_train(X_001_test, train, "FOREST")
    y_n_pred = np.vstack((y_n_pred, y_sing_i_pred))

y_multi_pred_10 = np.mean(y_n_pred, axis=0)
score_func(y_true, y_multi_pred_10)
plot_results(y_true, y_multi_pred_10)


#XGB modeli
xgb = train_models(train_df_1, model="XGB")
y_xgb_pred = xgb.predict(X_001_test)
XGB_individual_scorelst = score_func(y_true, y_xgb_pred)
plot_results(y_true, y_xgb_pred)

y_single_xgb_pred = single_train(X_001_test, train, "XGB")
XGB_SingleTrain_scorelst = score_func(y_true, y_single_xgb_pred)
plot_results(y_true, y_single_xgb_pred)

N=5
y_n_pred = y_single_xgb_pred
for i in range(1,N):
    X_001_test = prepare_test_data(fd_001_test, n=i)
    y_single_i_pred = single_train(X_001_test, train, "XGB")
    y_n_pred = np.vstack((y_n_pred, y_single_i_pred))

y_5_pred_xgb = np.mean(y_n_pred, axis=0)
XGB_5avg_scorelst = score_func(y_true, y_5_pred_xgb)
plot_results(y_true, y_5_pred_xgb)


def Bar_Plots(RF_score_lst, XGB_score_lst, LSTM_score_lst=0):
    hue = ["mae", "rmse", "r2"]
    if LSTM_score_lst != 0:

        df = pd.DataFrame(zip(hue*3, ["RFRegrssor"]*3 + ["LSTM"]*3 + ["XGBRegressor"]*3, RF_score_lst+LSTM_score_lst+
                              XGB_score_lst), columns=["Parameters", "Models", "Scores"])


    else:
        print(list(zip(hue * 3, ["RFRegrssor"] * 3 + ["XGBRegressor"] * 3, RF_score_lst + XGB_score_lst)))
        print(list(zip(hue * 2, ["RFRegrssor"] * 3 + ["XGBRegressor"] * 3, RF_score_lst + XGB_score_lst)))
        df = pd.DataFrame(zip(hue * 3, ["RFRegrssor"] * 3 + ["XGBRegressor"] * 3, RF_score_lst + XGB_score_lst),
                          columns=["Parameters", "Models", "Scores"])
    print(df.head(15))
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Models", y="Scores", hue="Parameters", data=df)
    plt.show()


Bar_Plots(RF_individual_scorelst, XGB_individual_scorelst, LSTM_individual_scorelst)
Bar_Plots(RF_SingleTrain_scorelst, XGB_SingleTrain_scorelst)
Bar_Plots(RF_5avg_scorelst, XGB_5avg_scorelst)

compare = pd.DataFrame(list(zip(y_true, y_pred, y_single_pred,y_multi_pred,y_multi_pred_10,y_xgb_pred,y_single_xgb_pred)),
               columns =['True', 'Forest_Predicted', 'Forest_Single_predicted', 'multi_5', 'multi_10', 'XGBoost','XGBoost_single'])
compare['unit_number'] = compare.index + 1

compare['Predicted_error'] = compare['True'] - compare['Forest_Predicted']
compare['Single_pred_error'] = compare['True'] - compare['Forest_Single_predicted']
compare['multi_5_error'] = compare['True'] - compare['multi_5']
compare['multi_10_error'] = compare['True'] - compare['multi_10']
compare['xgb_error'] = compare['True'] - compare['XGBoost']
compare['xgb_single_error'] = compare['True'] - compare['XGBoost_single']
ax1 = compare.plot(subplots=True, sharex=True, figsize=(20,20))
plt.show()

TTF = 10
train_df_1["label"] = np.where(train_df_1["RUL"] <= TTF, 1, 0)

sns.scatterplot(data=train_df_1, x="Nc", y="T50", hue="label")
plt.title("Scatter pattern Nc or T50")
plt.show()

X_class = train_df_1.iloc[:, :14].to_numpy()
Y_class = train_df_1.iloc[:, 15:].to_numpy()
Y_class = np.ravel(Y_class)

ros = RandomOverSampler(random_state=0)
ros.fit(X_class, Y_class)
X_resampled, Y_resampled = ros.fit_resample(X_class, Y_class)
print("The number of elements before random over sampling: ", len(X_class))
print("The number of classes before random over sampling: ", Counter(Y_class))
print("The number of elements after random over sampling: ", len(X_resampled))
print("The number of classes after random over sampling: ", Counter(Y_resampled))



X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=3)
forest = RandomForestClassifier(n_estimators=70, max_depth=8, random_state=193)
forest.fit(X_train, Y_train)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, Y_train)

def classificator_score(y_true, y_pred):
    print("Accuracy Score: ", round(accuracy_score(y_true, y_pred), 2))
    print("Precision Score: ", round(precision_score(y_true, y_pred), 2))
    print("Recall Score: ", round(recall_score(y_true, y_pred), 2))
    print("F1 Score: ", round(f1_score(y_true, y_pred), 2))
    return

classificator_score(Y_test, forest.predict(X_test))
y_xgb_pred = model_xgb.predict(X_001_test)
classificator_score(Y_test, model_xgb.predict(X_test))
print(test_new.head())
X_001_test = test_new.to_numpy()
a = RUL[[0]]
predicted = pd.DataFrame()
predicted["forest"] = forest.predict(X_001_test)
predicted["XGB"] = y_xgb_pred
predicted["RUL"] = RUL[0]
predicted["true_label"] = np.where(y_true <= TTF, 1, 0)
predicted["unit_number"] = predicted.index + 1

print(predicted.head())
print(predicted[predicted["true_label"] == 1])
print(predicted[predicted["forest"] != predicted["true_label"]])
print(predicted[predicted["XGB"] != predicted["true_label"]])

def expected_profit(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_true)):
        if (y_true[i] != y_pred[i]) and (y_true[i] == 1):
            FN+=1
        elif (y_true[i] != y_pred[i]) and (y_true[i] == 0):
            FP+=1
        elif (y_true[i] == y_pred[i]) and (y_true[i] == 0):
            TN+=1
        else:
            TP+=1
    print(f'TP ={TP}, TN = {TN}, FP = {FP}, FN = {FN}')
    print(f'expected profit {(300 * TP - 200 * FN - 100 * FP) * 1000}')
    return

def confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(5,5))
    sns.heatmap(sklearn.metrics.confusion_matrix(y_true, y_pred), annot=True, fmt=".5g")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

y_true_class = np.where(y_true <= TTF, 1, 0)
y_pred_class = predicted["forest"].to_list()

expected_profit(y_true_class, y_pred_class)
confusion_matrix(y_true_class, y_pred_class)

expected_profit(y_true_class, y_xgb_pred)
confusion_matrix(y_true_class, y_xgb_pred)

fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y_true_class, y_xgb_pred)
fpr_RF, tpr_RF, _ = metrics.roc_curve(y_true_class, y_pred_class)
auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)
auc_RF = metrics.auc(fpr_RF, tpr_RF)

plt.figure(figsize=(10,6))
plt.plot(fpr_xgb, tpr_xgb, label='ROC curve of XGB(area = %0.2f)' % auc_xgb)
plt.plot(fpr_RF, tpr_RF, label='ROC curve of RF(area = %0.2f)' % auc_RF)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, \
    GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# Transformer Encoder Layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dense1 = Dense(ff_dim, activation='relu')
        self.dense2 = Dense(head_size, activation='relu')
        self.projection = Dense(head_size)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        self.projection = Dense(input_shape[-1])  # Ensure projection layer matches input shape

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)
        dense_output = self.dense2(self.dense1(out1))
        dense_output = self.projection(dense_output)
        out2 = self.norm2(out1 + dense_output)
        return self.dropout(out2)


# Transformer Modeli
def build_transformer_model(input_shape, head_size=16, num_heads=2, ff_dim=64, num_blocks=4, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


# Veriyi yükle
X_train, X_test, y_train, y_test = train_test_split(seq_array, label_array, test_size=0.2, random_state=42)

# Transformer Modelini Eğit
input_shape = X_train.shape[1:]
transformer_model = build_transformer_model(input_shape, head_size=input_shape[-1])
transformer_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Transformer Tahminleri
transformer_preds = transformer_model.predict(X_test)

# XGBoost Eğitimi
import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
xgb_preds = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))

# LSTM Modelinden Tahmin Al
lstm_preds = model.predict(X_test)

# Ensemble (Ağırlıklı Ortalama)
ensemble_preds = (0.4 * transformer_preds.flatten() + 0.3 * lstm_preds.flatten() + 0.3 * xgb_preds)

# Sonuçları Değerlendir
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, ensemble_preds)
rmse = mean_squared_error(y_test, ensemble_preds)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2 = r2_score(y_test, ensemble_preds)
print(f"Ensemble Model MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")

transformer_pred_for_all = transformer_model.predict(seq_array_test)
xgb_pred_for_all = xgb_model.predict(seq_array_test.reshape(seq_array_test.shape[0], -1))

ensemble_preds_for_all = (0.4 * transformer_pred_for_all.flatten() + 0.3 * y_pred_test.flatten() + 0.3 * xgb_pred_for_all)
mae = mean_absolute_error(y_true_test, ensemble_preds_for_all)
rmse = mean_squared_error(y_true_test, ensemble_preds_for_all)
r2 = r2_score(y_true_test, ensemble_preds_for_all)
print(f"Ensemble Model for all test MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")
print(r2_score(y_true_test, xgb_pred_for_all))
print(r2_score(y_true_test, transformer_pred_for_all))
print(r2_score(y_true_test, y_pred_test))


