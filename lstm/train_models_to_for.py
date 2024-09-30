import pickle
import tensorflow
import pandas as pd
from preprocessing import get_nn_data
from train_models import train_dl_model
from mosqlient.models.score import Scorer
from lstm import build_model, build_lstm, build_lstm_multi, apply_model

end_date = '2024-08-11'

HIDDEN = 64
LOOK_BACK = 12
PREDICT_N = 3

%%time
#['BA', 'CE', 'PI', 'SP', 'MG', 'PA', 'MT', 'SC']
for state in ['SC']: 
    print(state)
    FILENAME_DATA = f'../data/dengue_{state}.csv.gz'

    df_ = pd.read_csv(FILENAME_DATA)

    feat = df_.shape[1]-1

    model = build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=4, loss='mse')


    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])
        
    train_dl_model(model, state, doenca='dengue',
                    end_date_train=None,
                    ratio = 1,
                    ini_date = '2015-01-01',
                    end_date = end_date,
                    plot=False, filename_data=FILENAME_DATA,
                    min_delta=0.001, label='for_base',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)
    

#['AL', 'MA', 'PB', 'PE', 'SE', 'RN', 'RJ', 'AM', 'AP', 'TO', 'RR',
#       'RO', 'AC', 'DF', 'GO', 'MS', 'RS', 'PR']

for state in ['RS', 'PR']:

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
                    end_date = end_date,
                    plot=False, filename_data=FILENAME_DATA,
                    min_delta=0.001, label='for_att',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)
    

