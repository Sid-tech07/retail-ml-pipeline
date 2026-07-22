"""
tests/test_pipeline.py
Complete pytest test suite
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')


# ── Fixtures ──────────────────────────────────────

@pytest.fixture
def raw_data():
    df1 = pd.read_excel(
        'online_retail_II.xlsx',
        sheet_name='Year 2009-2010',
        nrows=1000
    )
    return df1


@pytest.fixture
def clean_data():
    return pd.read_csv(
        'data/clean_transactions.csv'
    )


@pytest.fixture
def customer_features():
    return pd.read_csv(
        'data/customer_features.csv'
    )


@pytest.fixture
def trained_model():
    model  = joblib.load('models/rf_regressor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/feature_cols.json') as f:
        features = json.load(f)
    return model, scaler, features


@pytest.fixture
def metrics():
    with open('models/metrics.json') as f:
        return json.load(f)


# ── Group 1: Data Files ───────────────────────────

class TestDataFiles:

    def test_raw_data_exists(self):
        assert os.path.exists(
            'online_retail_II.xlsx'
        )

    def test_clean_data_exists(self):
        assert os.path.exists(
            'data/clean_transactions.csv'
        )

    def test_customer_features_exists(self):
        assert os.path.exists(
            'data/customer_features.csv'
        )

    def test_model_files_exist(self):
        files = [
            'models/rf_regressor.pkl',
            'models/scaler.pkl',
            'models/metrics.json',
            'models/feature_cols.json',
            'models/training_stats.json'
        ]
        for f in files:
            assert os.path.exists(f), \
                f"Missing: {f}"

    def test_output_dir_exists(self):
        os.makedirs('outputs', exist_ok=True)
        assert os.path.exists('outputs')


# ── Group 2: Data Quality ─────────────────────────

class TestDataQuality:

    def test_no_null_values(self, clean_data):
        null_count = clean_data.isnull().sum().sum()
        assert null_count == 0, \
            f"Found {null_count} nulls!"

    def test_no_negative_quantity(self, clean_data):
        neg = (clean_data['Quantity'] <= 0).sum()
        assert neg == 0, \
            f"Found {neg} negative quantities!"

    def test_no_negative_price(self, clean_data):
        neg = (clean_data['Price'] <= 0).sum()
        assert neg == 0, \
            f"Found {neg} negative prices!"

    def test_no_negative_amount(self, clean_data):
        neg = (clean_data['amount'] <= 0).sum()
        assert neg == 0, \
            f"Found {neg} negative amounts!"

    def test_no_cancelled_orders(self, clean_data):
        cancelled = clean_data['Invoice'].astype(
            str
        ).str.startswith('C').sum()
        assert cancelled == 0, \
            f"Found {cancelled} cancelled orders!"

    def test_customer_id_not_null(self, clean_data):
        nulls = clean_data[
            'Customer ID'
        ].isnull().sum()
        assert nulls == 0

    def test_customer_id_positive(self, clean_data):
        neg = (clean_data['Customer ID'] <= 0).sum()
        assert neg == 0

    def test_minimum_rows(self, clean_data):
        assert len(clean_data) >= 500000, \
            f"Too few rows: {len(clean_data)}"

    def test_required_columns(self, clean_data):
        required = [
            'Invoice', 'Customer ID',
            'Quantity', 'Price', 'amount',
            'InvoiceDate', 'Country'
        ]
        for col in required:
            assert col in clean_data.columns, \
                f"Missing column: {col}"

    def test_uk_only(self, clean_data):
        countries = clean_data['Country'].unique()
        assert len(countries) == 1
        assert countries[0] == 'United Kingdom'

    def test_date_range_valid(self, clean_data):
        dates = pd.to_datetime(
            clean_data['InvoiceDate']
        )
        assert dates.min() >= pd.Timestamp(
            '2009-01-01'
        )
        assert dates.max() <= pd.Timestamp(
            '2012-01-01'
        )


# ── Group 3: Feature Engineering ─────────────────

class TestFeatureEngineering:

    def test_no_null_features(
        self, customer_features
    ):
        with open('models/feature_cols.json') as f:
            feature_cols = json.load(f)
        nulls = customer_features[
            feature_cols
        ].isnull().sum().sum()
        assert nulls == 0

    def test_target_positive(
        self, customer_features
    ):
        neg = (
            customer_features['total_spend'] <= 0
        ).sum()
        assert neg == 0

    def test_minimum_customers(
        self, customer_features
    ):
        assert len(customer_features) >= 1000

    def test_rfm_scores_valid(
        self, customer_features
    ):
        for col in ['r_score', 'f_score', 'm_score']:
            if col in customer_features.columns:
                assert customer_features[col].min() >= 1
                assert customer_features[col].max() <= 5

    def test_ratios_between_0_and_1(
        self, customer_features
    ):
        ratio_cols = [
            'weekend_ratio',
            'high_value_ratio',
            'q4_spend_ratio'
        ]
        for col in ratio_cols:
            if col in customer_features.columns:
                assert customer_features[col].min() >= 0
                assert customer_features[col].max() <= 1

    def test_days_since_purchase_positive(
        self, customer_features
    ):
        col = 'days_since_last_purchase'
        if col in customer_features.columns:
            neg = (
                customer_features[col] < 0
            ).sum()
            assert neg == 0

    def test_feature_count(
        self, customer_features
    ):
        with open('models/feature_cols.json') as f:
            feature_cols = json.load(f)
        assert len(feature_cols) >= 10


# ── Group 4: Model ────────────────────────────────

class TestModel:

    def test_model_loads(self):
        model  = joblib.load(
            'models/rf_regressor.pkl'
        )
        scaler = joblib.load('models/scaler.pkl')
        assert model  is not None
        assert scaler is not None

    def test_model_has_predict(self):
        model = joblib.load(
            'models/rf_regressor.pkl'
        )
        assert hasattr(model, 'predict')

    def test_prediction_positive(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(50)
        X_scaled = scaler.transform(X)
        preds    = np.expm1(
            model.predict(X_scaled)
        )
        assert (preds > 0).all()

    def test_prediction_reasonable_range(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(100)
        X_scaled = scaler.transform(X)
        preds    = np.expm1(
            model.predict(X_scaled)
        )
        assert preds.min() >= 1
        assert preds.max() <= 500000

    def test_feature_count_matches(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(10)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        assert len(preds) == 10

    def test_model_consistent(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(5)
        X_scaled = scaler.transform(X)
        pred1 = model.predict(X_scaled)
        pred2 = model.predict(X_scaled)
        np.testing.assert_allclose(
            pred1, pred2,
            rtol=1e-10,
            err_msg="Predictions not consistent!"
        )

    def test_scaler_consistent(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(5)
        scaled1 = scaler.transform(X)
        scaled2 = scaler.transform(X)
        np.testing.assert_array_equal(
            scaled1, scaled2
        )


# ── Group 5: Metrics ──────────────────────────────

class TestMetrics:

    def test_r2_above_threshold(self, metrics):
        r2 = metrics['r2']
        assert r2 > 0.75, \
            f"R² {r2} below 0.75!"

    def test_mae_below_threshold(self, metrics):
        mae = metrics['mae']
        assert mae < 600, \
            f"MAE £{mae:,.0f} above £600!"

    def test_rmse_below_threshold(self, metrics):
        rmse = metrics['rmse']
        assert rmse < 4000, \
            f"RMSE £{rmse:,.0f} above £4000!"

    def test_mape_below_threshold(self, metrics):
        mape = metrics['mape']
        assert mape < 15.0, \
            f"MAPE {mape:.1f}% above 15%!"

    def test_metrics_file_complete(self, metrics):
        required = ['mae', 'rmse', 'r2', 'mape']
        for field in required:
            assert field in metrics

    def test_log_transform_flag(self, metrics):
        assert metrics.get('log_transform') == True

    def test_features_recorded(self, metrics):
        assert 'features_used' in metrics
        assert len(metrics['features_used']) >= 10


# ── Group 6: Prediction Pipeline ──────────────────

class TestPredictionPipeline:

    def test_single_prediction(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        customer = customer_features[
            features
        ].iloc[0].to_dict()
        X = pd.DataFrame([customer])
        X_scaled  = scaler.transform(X)
        pred_log  = model.predict(X_scaled)[0]
        pred_amt  = np.expm1(pred_log)
        assert pred_amt > 0
        assert not np.isnan(pred_amt)
        assert not np.isinf(pred_amt)

    def test_batch_prediction(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(100)
        X_scaled = scaler.transform(X)
        preds    = np.expm1(
            model.predict(X_scaled)
        )
        assert len(preds) == 100
        assert (preds > 0).all()

    def test_segment_classification(
        self, trained_model, customer_features
    ):
        model, scaler, features = trained_model
        X = customer_features[features].head(50)
        X_scaled = scaler.transform(X)
        preds    = np.expm1(
            model.predict(X_scaled)
        )

        def get_segment(spend):
            if spend >= 5000:
                return 'VIP'
            elif spend >= 2000:
                return 'High Value'
            elif spend >= 500:
                return 'Medium Value'
            else:
                return 'Standard'

        valid    = [
            'VIP', 'High Value',
            'Medium Value', 'Standard'
        ]
        segments = [get_segment(p) for p in preds]
        for seg in segments:
            assert seg in valid

    def test_prediction_vs_actual(
        self, trained_model, customer_features
    ):
        from sklearn.metrics import r2_score
        from sklearn.model_selection import (
            train_test_split
        )
        model, scaler, features = trained_model
        X = customer_features[features]
        y = np.log1p(
            customer_features['total_spend']
        )
        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        X_test_s = scaler.transform(X_test)
        y_pred   = model.predict(X_test_s)
        r2       = r2_score(y_test, y_pred)
        assert r2 > 0.70, \
            f"R² {r2:.4f} too low!"


# ── Group 7: Training Stats ───────────────────────

class TestTrainingStats:

    def test_training_stats_exist(self):
        assert os.path.exists(
            'models/training_stats.json'
        )

    def test_training_stats_complete(self):
        with open(
            'models/training_stats.json'
        ) as f:
            stats = json.load(f)
        with open(
            'models/feature_cols.json'
        ) as f:
            features = json.load(f)
        for feat in features:
            assert feat in stats, \
                f"Missing stats for: {feat}"

    def test_stats_have_required_fields(self):
        with open(
            'models/training_stats.json'
        ) as f:
            stats = json.load(f)
        required = [
            'mean', 'std', 'min',
            'max', 'median'
        ]
        for feat, feat_stats in stats.items():
            for field in required:
                assert field in feat_stats
                