from neuralprophet import NeuralProphet, set_log_level, plot_forecast_matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import warnings
import t1

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
set_log_level("ERROR")

# GPU 지정 (멀티 GPU 사용 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Loading Model...")
# 학습된 모델 불러오기
model_path = "./model/neupModel_nforecasts160.pt"
# NeuralProphet을 안전 목록에 추가
torch.serialization.add_safe_globals([NeuralProphet])
# 전체 모델 로드
m = torch.load(model_path)

# 예측 수행
print("Forecasting...")
df = m.make_future_dataframe(t1.df, n_historic_predictions=True)
forecast = m.predict(df)
m.set_plotting_backend("matplotlib")

#df_zoomed = forecast[150:350]
df_zoomed = forecast[49900:]
plot_forecast_matplotlib.plot(
    fcst=df_zoomed,
    quantiles=[],
    xlabel="Time",
    ylabel='Surge',
    highlight_forecast=8, # 특정 yhat 선택
)
plt.savefig("forecast_yhat8_end.png", dpi=300)

print("Complete!")
#print(forecast.to_string())

# yhat1
df_residuals = pd.DataFrame({"ds": df["ds"], "residuals": df["y"] - forecast["yhat1"]})
df_residuals_limited = df_residuals.iloc[:49000]
plt.figure(figsize=(12, 6))
plt.plot(df_residuals_limited["ds"], df_residuals_limited["residuals"], label="Residuals yhat1", color="b")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.grid()
plt.title("Residuals for yhat1")
plt.savefig("residuals_yhat1.png", dpi=300)
print("Residuals of yhat1 saved")
# yhat160
df_residuals = pd.DataFrame({"ds": df["ds"], "residuals": df["y"] - forecast["yhat160"]})
df_residuals_limited = df_residuals.iloc[:49000]
plt.plot(df_residuals_limited["ds"], df_residuals_limited["residuals"], label="Residuals yhat160", color="b")
plt.title("Residuals for yhat160")
plt.savefig("residuals_yhat160.png", dpi=300)
print("Residuals of yhat160 saved")

residuals = {}
for i in range(1, 161):  # yhat1 ~ yhat160
    col_name = f"yhat{i}"
    residuals[col_name] = (df["y"] - forecast[col_name]).abs().mean()  # 평균 절대 오차(MAE)
# DF
df_residuals_summary = pd.DataFrame(list(residuals.items()), columns=["yhat", "Mean Absolute Error"])
# 잔차가 작은 순서대로 정렬
df_residuals_summary = df_residuals_summary.sort_values(by="Mean Absolute Error")
print(df_residuals_summary.to_string())