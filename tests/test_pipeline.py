import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")


@pytest.fixture
def clean_data():
    return pd.read_csv("data/clean_transactions.csv")

@pytest.fixture
def customer_features():
    return pd.read_csv("data/customer_features.csv")

@pytest.fixture
def trained_model():
    model  = joblib.load("models/rf_regressor.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_cols.json") as f:
        features = json.load(f)
    return model, scaler, features

@pytest.fixture
def metrics():
    with open("models/metrics.json") as f:
        return json.load(f)


class TestDataFiles:

    def test_raw_data_exists(self):
        if not os.path.exists("online_retail_II.xlsx"):
            pytest.skip("Raw data not in CI/CD")
        assert os.path.exists("online_retail_II.xlsx")

    def test_clean_data_exists(self):
        assert os.path.exists("data/clean_transactions.csv")

    def test_customer_features_exists(self):
        assert os.path.exists("data/customer_features.csv")

    def test_model_files_exist(self):
        files = [
            "models/rf_regressor.pkl",
            "models/scaler.pkl",
            "models/metrics.json",
            "models/feature_cols.json",
            "models/training_stats.json"
        ]
        for f in files:
            assert os.path.exists(f), f"Missing: {f}"

    def test_output_dir_exists(self):
        os.makedirs("outputs", exist_ok=True)
        assert os.path.exists("outputs")


class TestDataQuality:

    def test_no_null_values(self, clean_data):
        assert clean_data.isnull().sum().sum() == 0

    def test_no_negative_quantity(self, clean_data):
        assert (clean_data["Quantity"] <= 0).sum() == 0

    def test_no_negative_price(self, clean_data):
        assert (clean_data["Price"] <= 0).sum() == 0

    def test_no_negative_amount(self, clean_data):
        assert (clean_data["amount"] <= 0).sum() == 0

    def test_no_cancelled_orders(self, clean_data):
        cancelled = clean_data["Invoice"].astype(str).str.startswith("C").sum()
        assert cancelled == 0

    def test_customer_id_not_null(self, clean_data):
        assert clean_data["Customer ID"].isnull().sum() == 0

    def test_customer_id_positive(self, clean_data):
        assert (clean_data["Customer ID"] <= 0).sum() == 0

    def test_minimum_rows(self, clean_data):
        assert len(clean_data) >= 1000

    def test_required_columns(self, clean_data):
        required = ["Invoice", "Customer ID", "Quantity", "Price", "amount", "InvoiceDate", "Country"]
        for col in required:
            assert col in clean_data.columns

    def test_uk_only(self, clean_data):
        countries = clean_data["Country"].unique()
        assert len(countries) == 1
        assert countries[0] == "United Kingdom"

    def test_date_range_valid(self, clean_data):
        dates = pd.to_datetime(clean_data["InvoiceDate"])
        assert dates.min() >= pd.Timestamp("2009-01-01")
        assert dates.max() <= pd.Timestamp("2012-01-01")


class TestFeatureEngineering:

    def test_no_null_features(self, customer_features):
        with open("models/feature_cols.json") as f:
            feature_cols = json.load(f)
        nulls = customer_features[feature_cols].isnull().sum().sum()
        assert nulls == 0

    def test_target_positive(self, customer_features):
        assert (customer_features["total_spend"] <= 0).sum() == 0

    def test_minimum_customers(self, customer_features):
        assert len(customer_features) >= 100

    def test_ratios_between_0_and_1(self, customer_features):
        ratio_cols = ["weekend_ratio", "high_value_ratio", "q4_spend_ratio"]
        for col in ratio_cols:
            if col in customer_features.columns:
                assert customer_features[col].min() >= 0
                assert customer_features[col].max() <= 1

    def test_days_since_purchase_positive(self, customer_features):
        col = "days_since_last_purchase"
        if col in customer_features.columns:
            assert (customer_features[col] < 0).sum() == 0

    def test_feature_count(self, customer_features):
        with open("models/feature_cols.json") as f:
            feature_cols = json.load(f)
        assert len(feature_cols) >= 10


class TestModel:

    def test_model_loads(self):
        model  = joblib.load("models/rf_regressor.pkl")
        scaler = joblib.load("models/scaler.pkl")
        assert model  is not None
        assert scaler is not None

    def test_model_has_predict(self):
        model = joblib.load("models/rf_regressor.pkl")
        assert hasattr(model, "predict")

    def test_prediction_positive(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(50)
        X_s   = scaler.transform(X)
        preds = np.expm1(model.predict(X_s))
        assert (preds > 0).all()

    def test_prediction_reasonable_range(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(100)
        X_s   = scaler.transform(X)
        preds = np.expm1(model.predict(X_s))
        assert preds.min() >= 1
        assert preds.max() <= 500000

    def test_feature_count_matches(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(10)
        X_s   = scaler.transform(X)
        preds = model.predict(X_s)
        assert len(preds) == 10

    def test_model_consistent(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(5)
        X_s   = scaler.transform(X)
        pred1 = model.predict(X_s)
        pred2 = model.predict(X_s)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)

    def test_scaler_consistent(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X       = customer_features[features].head(5)
        scaled1 = scaler.transform(X)
        scaled2 = scaler.transform(X)
        np.testing.assert_array_equal(scaled1, scaled2)


class TestMetrics:

    def test_r2_above_threshold(self, metrics):
        r2          = metrics["r2"]
        model_name  = metrics.get("model_name", "")
        if "CI/CD" in model_name:
            pytest.skip("CI/CD sample model - skip")
        assert r2 > 0.75, f"R2 {r2} below 0.75!"

    def test_mae_below_threshold(self, metrics):
        assert metrics["mae"] < 5000

    def test_rmse_below_threshold(self, metrics):
        assert metrics["rmse"] < 10000

    def test_mape_below_threshold(self, metrics):
        mape       = metrics["mape"]
        model_name = metrics.get("model_name", "")
        if "CI/CD" in model_name:
            pytest.skip("CI/CD sample model - skip")
        assert mape < 20.0, f"MAPE {mape} too high!"

    def test_metrics_file_complete(self, metrics):
        for field in ["mae", "rmse", "r2", "mape"]:
            assert field in metrics

    def test_log_transform_flag(self, metrics):
        assert metrics.get("log_transform") == True

    def test_features_recorded(self, metrics):
        assert "features_used" in metrics
        assert len(metrics["features_used"]) >= 10


class TestPredictionPipeline:

    def test_single_prediction(self, trained_model, customer_features):
        model, scaler, features = trained_model
        customer = customer_features[features].iloc[0].to_dict()
        X        = pd.DataFrame([customer])
        X_s      = scaler.transform(X)
        pred     = np.expm1(model.predict(X_s)[0])
        assert pred > 0
        assert not np.isnan(pred)
        assert not np.isinf(pred)

    def test_batch_prediction(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(100)
        X_s   = scaler.transform(X)
        preds = np.expm1(model.predict(X_s))
        assert len(preds) == 100
        assert (preds > 0).all()

    def test_segment_classification(self, trained_model, customer_features):
        model, scaler, features = trained_model
        X     = customer_features[features].head(50)
        X_s   = scaler.transform(X)
        preds = np.expm1(model.predict(X_s))

        def get_segment(spend):
            if spend >= 5000:   return "VIP"
            elif spend >= 2000: return "High Value"
            elif spend >= 500:  return "Medium Value"
            else:               return "Standard"

        valid    = ["VIP", "High Value", "Medium Value", "Standard"]
        segments = [get_segment(p) for p in preds]
        for seg in segments:
            assert seg in valid

    def test_prediction_vs_actual(self, trained_model, customer_features):
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        model, scaler, features = trained_model
        X  = customer_features[features]
        y  = np.log1p(customer_features["total_spend"])
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_te_s = scaler.transform(X_te)
        r2     = r2_score(y_te, model.predict(X_te_s))
        assert r2 > 0.0, f"R2 {r2:.4f} too low!"


class TestTrainingStats:

    def test_training_stats_exist(self):
        assert os.path.exists("models/training_stats.json")

    def test_training_stats_complete(self):
        with open("models/training_stats.json") as f:
            stats = json.load(f)
        with open("models/feature_cols.json") as f:
            features = json.load(f)
        for feat in features:
            assert feat in stats

    def test_stats_have_required_fields(self):
        with open("models/training_stats.json") as f:
            stats = json.load(f)
        required = ["mean", "std", "min", "max", "median"]
        for feat, feat_stats in stats.items():
            for field in required:
                assert field in feat_stats
