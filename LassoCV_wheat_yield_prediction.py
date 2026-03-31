import pandas as pd
import numpy as np
df = pd.read_excel("/content/Wheat_Punjab_Rabi.xlsx")
df.shape

df

df.corr(method="spearman")

x_train = df[df.year < 2019]
x_test  = df[df.year >= 2019]
y_train = x_train["yield"]
y_test = x_test["yield"]

corr_column=["total_rainfall_curr_11","total_rainfall_curr_12","max_temperature_curr_10","min_temperature_curr_11"]
x_train = x_train[corr_column]
x_test = x_test[corr_column]


x_train["mean_rainfall"] = (x_train["total_rainfall_curr_11"] + x_train["total_rainfall_curr_12"]) / 2
x_test["mean_rainfall"] = (x_test["total_rainfall_curr_11"] + x_test["total_rainfall_curr_12"]) / 2

x_train = x_train.drop(columns=["total_rainfall_curr_11","total_rainfall_curr_12"])
x_test = x_test.drop(columns=["total_rainfall_curr_11","total_rainfall_curr_12"])
x_test

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LassoCV,Lasso
from sklearn.preprocessing import PowerTransformer, RobustScaler


pt = PowerTransformer(method='yeo-johnson')
x_train = pt.fit_transform(x_train)
x_test = pt.transform(x_test)

rsc = RobustScaler()

x_train_scaled = rsc.fit_transform(x_train)
x_test_scaled = rsc.transform(x_test)

print(pd.DataFrame(x_test_scaled).describe())

results = []

model = LassoCV(cv=4).fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

results.append({
    "alpha": model.alpha_,
    "MAE":  round(mae, 4),
    "RMSE": round(rmse, 4),
})

import pandas as pd
results1 = pd.DataFrame(results)
print(results1)

import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE:{mape:.2f}")

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})

alt.Chart(metrics_df).mark_bar().encode(
    x="Metric:N",
    y="Value:Q",
    tooltip=["Value"]
).properties(
    title="MAE vs RMSE"
)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

df_plot = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

df_plot["Residual"] = df_plot["Actual"] - df_plot["Predicted"]

import plotly.graph_objects as go

fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=df_plot["Actual"],
    y=df_plot["Predicted"],
    mode="markers",
    name="Predictions"
))

fig_pred.add_trace(go.Scatter(
    x=df_plot["Actual"],
    y=df_plot["Actual"],
    mode="lines",
    name="Perfect Prediction"
))

fig_pred.update_layout(
    title=f"Prediction Error Plot<br>MAE={mae:.3f}, RMSE={rmse:.3f}",
    xaxis_title="Actual",
    yaxis_title="Predicted"
)

fig_pred.show()

import numpy as np
import pandas as pd


alphas = model.alphas_


mse_path = model.mse_path_


mean_mse = mse_path.mean(axis=1)


rmse = np.sqrt(mean_mse)

df_alpha = pd.DataFrame({
    "alpha": alphas,
    "MSE": mean_mse,
    "RMSE": rmse
})

import plotly.express as px

fig = px.line(
    df_alpha,
    x="alpha",
    y="RMSE",
    log_x=True,
    markers=True,
    title="LassoCV: Alpha vs RMSE"
)

fig.show()
