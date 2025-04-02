import os, joblib


def save_model(model, model_dir, model_name):
    joblib.dump(model, os.path.join(model_dir, model_name))
