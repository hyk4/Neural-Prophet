from neuralprophet import NeuralProphet, set_log_level
import pandas as pd
import torch
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")

# GPU 지정 (멀티 GPU 사용 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 데이터 로드 & 전처리
file_path = "./time_series/orgin_dataset/LIN_001_IRR_MOT.mot"
df = pd.read_csv(file_path, skiprows=3, delim_whitespace=True)
df.columns = ["Time", "Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
df = df.apply(pd.to_numeric, errors='coerce')
variables_to_plot = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]

# 모델 저장 경로
model_path = "./model/neupModel_nforecasts160.pt"

# 데이터 전처리
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df = df.rename(columns={'Time': 'ds', 'Surge': 'y'})

# Train-Validation Split
df_train, df_val = NeuralProphet().split_df(df, valid_p=0.2, freq='0.1s')

# 모델 정의
def create_model():
    m = NeuralProphet(
        n_lags=160,                # 과거 16초 데이터로 학습: Auto-Regression
        n_forecasts=160,           # 미래 16초 예측
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        batch_size=64,
        ar_layers=[32, 16],
        lagged_reg_layers=[16, 8],
    )
    m.add_seasonality(name='16sec_cycle', period=160, fourier_order=3)
    m.set_plotting_backend("plotly-static")

    # Lagged Regressors
    for var in ["Sway", "Heave", "Roll", "Pitch", "Yaw"]:
        m.add_lagged_regressor(var)

    return m

if __name__ == "__main__":
    print("Training new model...")
    m = create_model()
    metrics = m.fit(df_train, validation_df=df_val, freq='0.1s')
    print(metrics)

    torch.save(m, model_path)
    print(f"Model saved: {model_path}")