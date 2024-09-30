import pickle
import tensorflow
import pandas as pd
from preprocessing import get_nn_data
from train_models import train_dl_model
from mosqlient.models.score import Scorer
from lstm import build_model, build_lstm, build_lstm_multi, apply_model


states_BR = ['AL',
 'BA',
 'CE',
 'MA',
 'PB',
 'PE',
 'PI',
 'SE',
 'RN',
 'SP',
 'MG',
 'RJ',
 'ES',
 'AM',
 'AP',
 'TO',
 'RR',
 'RO',
 'AC',
 'PA',
 'DF',
 'GO',
 'MT',
 'MS',
 'RS',
 'SC',
 'PR']


end_date = '2024-08-04'

HIDDEN = 64
LOOK_BACK = 12
PREDICT_N = 3

for state in states_BR: 
    print(state)
    FILENAME_DATA = f'../data/dengue_{state}.csv.gz'

    df_ = pd.read_csv(FILENAME_DATA)

    feat = df_.shape[1]-1
    
    model = build_lstm_multi(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=4, loss='mse')

    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])
        
    train_dl_model(model, state, doenca='dengue',
                    end_date_train=None,
                    ratio = 1,
                    ini_date = '2015-01-01',
                    end_date = '2022-09-04',
                    plot=False, filename_data=FILENAME_DATA,
                    min_delta=0.001, label='att',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)
    

    metrics1, metrics2 = apply_model(state, ini_date = '2015-06-01', 
                        end_date = end_date, look_back = LOOK_BACK, end_train_date = '2022-09-04', batch_size = 4, 
                        predict_n = PREDICT_N,  ratio = None,
                        label_pred= f'dengue_att',
                        model_name = f'trained_{state}_dengue_att', 
                        filename = f'../data/dengue_{state}.csv.gz', plot = True)
    
    
    


    model = build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=4, loss='mse')

    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])

    train_dl_model(model, state, doenca='dengue',
                    end_date_train=None,
                    ratio = 1,
                    ini_date = '2015-01-01',
                    end_date = '2022-09-04',
                    plot=False, filename_data=FILENAME_DATA,
                    min_delta=0.001, label='base',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)

    metrics1, metrics2 = apply_model(state, ini_date = '2015-06-01', 
                        end_date = end_date, look_back = LOOK_BACK, end_train_date = '2022-09-04', batch_size = 4, 
                        predict_n = PREDICT_N,  ratio = None,
                        label_pred= f'dengue_base',
                        model_name = f'trained_{state}_dengue_base', 
                        filename = f'../data/dengue_{state}.csv.gz', plot = False)
    
    
