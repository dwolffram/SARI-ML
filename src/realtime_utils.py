import pandas as pd
from darts import TimeSeries, concatenate
import matplotlib.pyplot as plt


def load_target_series():
    target = pd.read_csv('https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/main/data/icosari/sari/target-icosari-sari.csv')

    ts_target = TimeSeries.from_group_dataframe(target, group_cols=['age_group'], 
                                             time_col='date', value_cols='value', 
                                             freq='7D', fillna_value=0)
    ts_target = concatenate(ts_target, axis=1)
    ts_target = ts_target.with_columns_renamed(ts_target.static_covariates.age_group.index, 'icosari-sari-' + ts_target.static_covariates.age_group)
    ts_target = ts_target.with_columns_renamed('icosari-sari-00+', 'icosari-sari-DE')
    
    return ts_target


def load_nowcast(forecast_date):
    filepath = f'../data/nowcasts/KIT-baseline/{forecast_date}-icosari-sari-KIT-baseline.csv'
    df = pd.read_csv(filepath)
    df = df[(df.type == 'quantile') & (df.horizon >= -3)]
    df = df.rename(columns={'target_end_date' : 'date'})
    
    all_nowcasts = []
    for age in df.age_group.unique():
        # print(age)
        df_temp = df[df.age_group == age]

        # transform nowcast into a TimeSeries object
        nowcast_age = TimeSeries.from_group_dataframe(df_temp, group_cols=['age_group', 'quantile'],
                              time_col='date', value_cols='value', 
                              freq='7D', fillna_value=0)

        nowcast_age = concatenate(nowcast_age, axis='sample')
        nowcast_age.static_covariates.drop(columns=['quantile'], inplace=True, errors='ignore')
        nowcast_age = nowcast_age.with_columns_renamed(nowcast_age.components, ['icosari-sari-' + age])

        all_nowcasts.append(nowcast_age)
        
    all_nowcasts = concatenate(all_nowcasts, axis='component')
    all_nowcasts = all_nowcasts.with_columns_renamed('icosari-sari-00+', 'icosari-sari-DE')
    
    return all_nowcasts


def make_target_paths(target_series, nowcast):
    # cut known truth series and append nowcasted values
    target_temp = target_series.drop_after(nowcast.start_time())
    
    # every entry is a multivariate timeseries (one sample path for each age group)
    # there is one entry per quantile level
    target_list = [concatenate([target_temp[age].append_values(nowcast[age].univariate_values(sample=i)) for age in nowcast.components], axis='component') 
                   for i in range(nowcast.n_samples)]
    
    return target_list
