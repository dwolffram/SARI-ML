import pandas as pd
from epiweeks import Week
from darts.metrics.metrics import mql

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

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

METRIC = [mql for _ in QUANTILES]
METRIC_KWARGS = [{'q': q} for q in QUANTILES]
