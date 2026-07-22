
"""
SageMaker inference script
Handles predictions for customer spend
"""

import joblib
import numpy as np
import json
import os

def model_fn(model_dir):
    model  = joblib.load(
        os.path.join(model_dir, 'rf_regressor.pkl')
    )
    scaler = joblib.load(
        os.path.join(model_dir, 'scaler.pkl')
    )
    with open(
        os.path.join(model_dir, 'feature_cols.json')
    ) as f:
        features = json.load(f)
    return {
        'model'   : model,
        'scaler'  : scaler,
        'features': features
    }

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['features'])
    raise ValueError(f"Unsupported: {content_type}")

def predict_fn(input_data, model_dict):
    model   = model_dict['model']
    scaler  = model_dict['scaler']
    scaled  = scaler.transform(
        input_data.reshape(1, -1)
    )
    pred_log = model.predict(scaled)[0]
    pred_amt = np.expm1(pred_log)

    if pred_amt >= 5000:
        segment = "VIP"
        action  = "Assign dedicated account manager"
    elif pred_amt >= 2000:
        segment = "High Value"
        action  = "Send premium offer"
    elif pred_amt >= 500:
        segment = "Medium Value"
        action  = "Send standard offer"
    else:
        segment = "Standard"
        action  = "Regular communication"

    return {
        'predicted_spend': round(float(pred_amt), 2),
        'segment'        : segment,
        'action'         : action
    }

def output_fn(prediction, accept):
    return json.dumps(prediction), accept
