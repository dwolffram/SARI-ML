import pandas as pd
from epiweeks import Week
from darts.metrics.metrics import mql

MODEL_NAMES = {
    'KIT-MeanEnsemble' : 'Ensemble',
    'lightgbm': 'LightGBM',
    'lightgbm_noCovariates': 'LightGBM-NoCovariates',
    'lightgbm_noCovid': 'LightGBM-NoCovid',
    'lightgbm_oracle': 'LightGBM-Oracle',
    'lightgbm_skip' : 'LightGBM-Skip',
    'tsmixer_covariates': 'TSMixer',
    'tsmixer': 'TSMixer-NoCovariates',
    'tsmixer_noCovid': 'TSMixer-NoCovid',
    'tsmixer_oracle': 'TSMixer-Oracle',
    'tsmixer_skip' : 'TSMixer-Skip',
    'KIT-hhh4' : 'hhh4-NoCovid',
    'KIT-hhh4_all_data': 'hhh4',
    'KIT-hhh4_all_data_oracle' : 'hhh4-Oracle',
    'KIT-hhh4_all_data_skip': 'hhh4-Skip',
    'KIT-baseline' : 'Nowcast',
    'KIT-persistence': 'Persistence',
    'baseline' : 'Historical'
}

# TRAIN_END        = pd.Timestamp('2018-09-30')
# VALIDATION_START = pd.Timestamp('2018-10-07')
# VALIDATION_END   = pd.Timestamp('2019-09-29')
# TEST_START       = pd.Timestamp('2019-10-06')
# TEST_END         = pd.Timestamp('2020-09-27')
EVAL_START       = pd.Timestamp('2023-01-01')

SEASON_DICT = {
    year: pd.to_datetime(Week(year + 1, 39, system="iso").enddate())
    for year in range(2014, 2020)
}

TARGETS = [
    'icosari-sari-DE',
    'icosari-sari-00-04',
    'icosari-sari-05-14',
    'icosari-sari-15-34',
    'icosari-sari-35-59',
    'icosari-sari-60-79',
    'icosari-sari-80+'
]

SOURCES = [
    'survstat', 
    'icosari', 
    'agi'
]

SOURCE_DICT = {
    'sari' : 'icosari',
    'are' : 'agi'
}

QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

METRIC = [mql for _ in QUANTILES]
METRIC_KWARGS = [{'q': q} for q in QUANTILES]
