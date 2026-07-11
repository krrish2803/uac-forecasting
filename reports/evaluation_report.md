# UAC Forecasting - Model Evaluation Report

## Model Comparison Summary

    target               model  MAE_1-day  RMSE_1-day  MAPE_1-day  MAE_7-day  RMSE_7-day  MAPE_7-day  MAE_14-day  RMSE_14-day  MAPE_14-day  MAE_overall  RMSE_overall  MAPE_overall
    in_hhs   Naive Persistence      11.00       11.00        0.45       6.43        7.15        0.26        6.36         7.19         0.26         6.36          7.19          0.26
    in_hhs Moving Average (7d)      21.43       21.43        0.88      27.14       28.40        1.11       42.21        45.39         1.71        42.21         45.39          1.71
    in_hhs              SARIMA      11.42       11.42        0.47      10.45       11.22        0.43       17.86        19.72         0.72        17.86         19.72          0.72
    in_hhs      Exp. Smoothing      16.68       16.68        0.68      72.87       77.97        2.98      108.95       117.90         4.42       108.95        117.90          4.42
    in_hhs       Random Forest      14.61       14.61        0.60       7.55        9.32        0.31        6.07         7.72         0.25         6.07          7.72          0.25
    in_hhs   Gradient Boosting       5.21        5.21        0.21      12.30       15.62        0.50        9.82        12.83         0.40         9.82         12.83          0.40
discharged   Naive Persistence       8.00        8.00      200.00       2.29        3.45       42.04        2.00         3.01        28.00         2.00          3.01         28.00
discharged Moving Average (7d)       6.57        6.57      164.29       2.43        3.03       41.00        2.89         3.39        34.18         2.89          3.39         34.18
discharged              SARIMA       9.64        9.64      241.05       3.57        4.54       59.12        3.77         4.45        46.16         3.77          4.45         46.16
discharged      Exp. Smoothing      10.44       10.44      261.07       5.56        6.19       82.69        4.68         5.32        59.87         4.68          5.32         59.87
discharged       Random Forest       5.72        5.72      142.95       2.96        3.42       45.33        2.09         2.60        28.20         2.09          2.60         28.20
discharged   Gradient Boosting       4.84        4.84      120.92       2.65        3.04       40.18        1.88         2.31        24.93         1.88          2.31         24.93


## Target: In Hhs

### Per-Horizon Metrics


#### Naive Persistence

- **1-day**: MAE=11.00, RMSE=11.00, MAPE=0.5%
- **7-day**: MAE=6.43, RMSE=7.15, MAPE=0.3%
- **14-day**: MAE=6.36, RMSE=7.19, MAPE=0.3%
- **overall**: MAE=6.36, RMSE=7.19, MAPE=0.3%

#### Moving Average (7d)

- **1-day**: MAE=21.43, RMSE=21.43, MAPE=0.9%
- **7-day**: MAE=27.14, RMSE=28.40, MAPE=1.1%
- **14-day**: MAE=42.21, RMSE=45.39, MAPE=1.7%
- **overall**: MAE=42.21, RMSE=45.39, MAPE=1.7%

#### SARIMA

- **1-day**: MAE=11.42, RMSE=11.42, MAPE=0.5%
- **7-day**: MAE=10.45, RMSE=11.22, MAPE=0.4%
- **14-day**: MAE=17.86, RMSE=19.72, MAPE=0.7%
- **overall**: MAE=17.86, RMSE=19.72, MAPE=0.7%

#### Exp. Smoothing

- **1-day**: MAE=16.68, RMSE=16.68, MAPE=0.7%
- **7-day**: MAE=72.87, RMSE=77.97, MAPE=3.0%
- **14-day**: MAE=108.95, RMSE=117.90, MAPE=4.4%
- **overall**: MAE=108.95, RMSE=117.90, MAPE=4.4%

#### Random Forest

- **1-day**: MAE=14.61, RMSE=14.61, MAPE=0.6%
- **7-day**: MAE=7.55, RMSE=9.32, MAPE=0.3%
- **14-day**: MAE=6.07, RMSE=7.72, MAPE=0.2%
- **overall**: MAE=6.07, RMSE=7.72, MAPE=0.2%

#### Gradient Boosting

- **1-day**: MAE=5.21, RMSE=5.21, MAPE=0.2%
- **7-day**: MAE=12.30, RMSE=15.62, MAPE=0.5%
- **14-day**: MAE=9.82, RMSE=12.83, MAPE=0.4%
- **overall**: MAE=9.82, RMSE=12.83, MAPE=0.4%

### Top 10 Features (Random Forest)

- in_hhs_lag1: 0.4957
- in_hhs_rmean7: 0.3141
- in_hhs_rmean14: 0.1673
- in_hhs_lag7: 0.0125
- transferred_out_rmean14: 0.0011
- transferred_out_rmean7: 0.0010
- apprehended_rmean7: 0.0010
- apprehended_rmean14: 0.0010
- in_cbp_rmean14: 0.0009
- apprehended_rstd14: 0.0006

### Top 10 Features (Gradient Boosting)

- in_hhs_lag1: 0.6321
- in_hhs_rmean7: 0.2799
- in_hhs_lag7: 0.0746
- in_hhs_rmean14: 0.0030
- apprehended_rmean7: 0.0027
- discharged_rmean14: 0.0019
- apprehended_rstd7: 0.0009
- in_hhs_lag14: 0.0009
- discharged_rmean7: 0.0006
- apprehended_lag1: 0.0006

## Target: Discharged

### Per-Horizon Metrics


#### Naive Persistence

- **1-day**: MAE=8.00, RMSE=8.00, MAPE=200.0%
- **7-day**: MAE=2.29, RMSE=3.45, MAPE=42.0%
- **14-day**: MAE=2.00, RMSE=3.01, MAPE=28.0%
- **overall**: MAE=2.00, RMSE=3.01, MAPE=28.0%

#### Moving Average (7d)

- **1-day**: MAE=6.57, RMSE=6.57, MAPE=164.3%
- **7-day**: MAE=2.43, RMSE=3.03, MAPE=41.0%
- **14-day**: MAE=2.89, RMSE=3.39, MAPE=34.2%
- **overall**: MAE=2.89, RMSE=3.39, MAPE=34.2%

#### SARIMA

- **1-day**: MAE=9.64, RMSE=9.64, MAPE=241.0%
- **7-day**: MAE=3.57, RMSE=4.54, MAPE=59.1%
- **14-day**: MAE=3.77, RMSE=4.45, MAPE=46.2%
- **overall**: MAE=3.77, RMSE=4.45, MAPE=46.2%

#### Exp. Smoothing

- **1-day**: MAE=10.44, RMSE=10.44, MAPE=261.1%
- **7-day**: MAE=5.56, RMSE=6.19, MAPE=82.7%
- **14-day**: MAE=4.68, RMSE=5.32, MAPE=59.9%
- **overall**: MAE=4.68, RMSE=5.32, MAPE=59.9%

#### Random Forest

- **1-day**: MAE=5.72, RMSE=5.72, MAPE=142.9%
- **7-day**: MAE=2.96, RMSE=3.42, MAPE=45.3%
- **14-day**: MAE=2.09, RMSE=2.60, MAPE=28.2%
- **overall**: MAE=2.09, RMSE=2.60, MAPE=28.2%

#### Gradient Boosting

- **1-day**: MAE=4.84, RMSE=4.84, MAPE=120.9%
- **7-day**: MAE=2.65, RMSE=3.04, MAPE=40.2%
- **14-day**: MAE=1.88, RMSE=2.31, MAPE=24.9%
- **overall**: MAE=1.88, RMSE=2.31, MAPE=24.9%

### Top 10 Features (Random Forest)

- discharged_lag1: 0.4182
- discharged_lag7: 0.2629
- discharged_rmean7: 0.1182
- net_pressure: 0.0374
- discharged_rmean14: 0.0266
- in_hhs_rmean7: 0.0254
- in_hhs_rmean14: 0.0188
- in_hhs_lag1: 0.0165
- in_hhs_lag7: 0.0121
- day_of_week: 0.0092

### Top 10 Features (Gradient Boosting)

- discharged_lag1: 0.3440
- discharged_lag7: 0.2900
- discharged_rmean7: 0.1784
- net_pressure: 0.0449
- day_of_week: 0.0221
- in_hhs_lag1: 0.0217
- in_hhs_rmean7: 0.0179
- apprehended_lag1: 0.0135
- discharged_lag14: 0.0106
- transferred_out_lag14: 0.0105