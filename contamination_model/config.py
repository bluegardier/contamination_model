from os import path

VERSION = "1.0.0"

# Paths
base_path = path.abspath("../")
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'model')
raw_data_path = path.join(data_path, 'raw')
processed_data_path = path.join(data_path, 'processed')


DF_PATH = raw_data_path + "/individuos_espec.csv"
DF_TARGET_PATH = raw_data_path + "/conexoes_espec.csv"
DF_TRAIN_PATH = processed_data_path + "/df_train.pickle"
DF_PREDICT_PATH = processed_data_path + "/df_predict.pickle"

to_drop_ld = [
    "taxi__V1",
    "taxi__V2",
    "familia",
    "visita_rara",
    "sem_filhos__V1",
    "sem_filhos__V2",
    "divorciado__V1",
    "divorciado__V2",
    "nao_estuda__V1",
    "nao_estuda__V2",
    "nao_trabalha__V1",
    "nao_trabalha__V2",
    "nao_pratica_esportes__V1",
    "nao_pratica_esportes__V2",
    "adequado__V1",
    "adequado__V2",
    "maior_65__V1",
    "maior_65__V2",
]

to_fillna_binary_var = [
    "estuda",
    "trabalha",
    "pratica_esportes",
]

models_list = [
    "lr",
    "lasso",
    "ridge",
    "en",
]

binary_variables = [
    "estuda",
    "trabalha",
    "pratica_esportes"
]
metric_list = [
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "RMSLE",
    "MAPE",
]

median_fill_variables = ["IMC", "idade"]
