import pandas as pd
from lstm import  build_lstm
import matplotlib.pyplot as plt 
from forecast import apply_forecast
from scipy.special import inv_boxcox 
from train_models import train_dl_model 

LOOK_BACK = 12
PREDICT_N = 3

code_to_state = {33: 'RJ', 32: 'ES', 41: 'PR', 23: 'CE', 21: 'MA',
 31: 'MG', 42: 'SC', 26: 'PE', 25: 'PB', 24: 'RN', 22: 'PI', 27: 'AL',
 28: 'SE', 35: 'SP', 43: 'RS', 15: 'PA', 16: 'AP', 14: 'RR',  11: 'RO',
 13: 'AM', 12: 'AC', 51: 'MT', 50: 'MS', 52: 'GO', 17: 'TO', 53: 'DF',
 29: 'BA'}

end_date = '2024-09-15'

for state in code_to_state.values():
    print(f'Forecasting: {state}')

    FILENAME_DATA = f'../data/dengue_{state}.csv.gz'
    df_ = pd.read_csv(FILENAME_DATA, index_col = 'date')

    feat = df_.shape[1]
    
    if state in ['BA', 'CE', 'PI', 'SP', 'MG', 'PA', 'MT', 'SC', 'ES']: 
        model_name = f'trained_{state}_dengue_for_base'

    elif state in ['AL', 'MA', 'PB', 'PE', 'SE', 'RN', 'RJ', 'AM', 'AP', 'TO', 'RR',
       'RO', 'AC', 'DF', 'GO', 'MS', 'RS', 'PR']:
        model_name = f'trained_{state}_dengue_for_att'

    print(model_name)

    df_for = apply_forecast(state, None, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                    filename=FILENAME_DATA, model_name=model_name)

    df_for.date = pd.to_datetime(df_for.date)

    df_for.to_csv(f'./forecast_tables/for_{state}.csv')

