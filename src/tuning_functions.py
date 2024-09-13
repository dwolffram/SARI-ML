import joblib
import numpy as np
from IPython.display import clear_output

def get_validation_chunks(test_year):
    validation_years = [year for year in SEASON_DICT if year < test_year][1:]
    validation_ends = [SEASON_DICT[year] for year in validation_years]
    train_ends = [SEASON_DICT[year - 1] for year in validation_years]
    return train_ends, validation_ends


def compute_validation_score(model, targets_train, targets_validation, covariates, start, horizon, num_samples, metric, metric_kwargs):
    model.fit(targets_train, past_covariates=covariates)
    
    scores = model.backtest(
        series=targets_validation,
        past_covariates=covariates,
        start=start,
        forecast_horizon=horizon,
        stride=1,
        last_points_only=False,
        retrain=False,
        verbose=True,
        num_samples=num_samples,
        metric=metric, 
        metric_kwargs=metric_kwargs
    )
    
    score = np.mean(scores) # WIS as mean of quantile scores
    
    return score if score != np.nan else float("inf")


def print_best_trial(study):
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
      
    
def run_trials(study_path, objective, n_trials, n_jobs):
    study = joblib.load(study_path)
    print_best_trial(study)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    joblib.dump(study, study_path)
    