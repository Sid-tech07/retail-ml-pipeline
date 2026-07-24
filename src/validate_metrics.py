import json
import sys
import os

def validate_metrics():

    print("="*55)
    print("CI/CD METRIC VALIDATION GATE")
    print("="*55)

    metrics_path = "models/metrics.json"
    if not os.path.exists(metrics_path):
        print("metrics.json not found!")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    mae        = metrics["mae"]
    rmse       = metrics["rmse"]
    r2         = metrics["r2"]
    mape       = metrics["mape"]
    model_name = metrics.get("model_name", "")

    print(f"\nLoaded metrics:")
    print(f"  Model : {model_name}")
    print(f"  MAE   : {mae:,.2f}")
    print(f"  RMSE  : {rmse:,.2f}")
    print(f"  R2    : {r2:.4f}")
    print(f"  MAPE  : {mape:.1f}%")

    # CI/CD sample model uses relaxed thresholds
    if "CI/CD" in model_name:
        print("\nCI/CD sample model detected")
        print("Using relaxed thresholds...")
        THRESHOLDS = {
            "max_mae" : 5000,
            "max_rmse": 10000,
            "min_r2"  : -1.0,
            "max_mape": 100.0,
        }
    else:
        print("\nProduction model detected")
        print("Using strict thresholds...")
        THRESHOLDS = {
            "max_mae" : 600,
            "max_rmse": 4000,
            "min_r2"  : 0.75,
            "max_mape": 15.0,
        }

    print(f"\nThresholds:")
    print(f"  MAE   < {THRESHOLDS['max_mae']:,}")
    print(f"  RMSE  < {THRESHOLDS['max_rmse']:,}")
    print(f"  R2    > {THRESHOLDS['min_r2']}")
    print(f"  MAPE  < {THRESHOLDS['max_mape']}%")

    validations = {
        "MAE check" : mae  < THRESHOLDS["max_mae"],
        "RMSE check": rmse < THRESHOLDS["max_rmse"],
        "R2 check"  : r2   > THRESHOLDS["min_r2"],
        "MAPE check": mape < THRESHOLDS["max_mape"],
    }

    print(f"\nValidation Results:")
    print("-"*45)

    all_passed = True
    for check, passed in validations.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {check:<12}: {status}")

    print("\n" + "="*55)
    if all_passed:
        print("ALL VALIDATIONS PASSED!")
        print("Model approved for deployment")
        print("="*55)
        sys.exit(0)
    else:
        failed = [k for k, v in validations.items() if not v]
        print("VALIDATION FAILED!")
        print(f"Failed: {failed}")
        print("="*55)
        sys.exit(1)


if __name__ == "__main__":
    validate_metrics()
