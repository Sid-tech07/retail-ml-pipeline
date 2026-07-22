"""
Step 5: Validate Metrics
CI/CD Gate — Pipeline stops if
model doesn't meet thresholds
"""

import json
import sys
import os

def validate_metrics():
    """
    CI/CD validation gate
    GitHub Actions calls this
    Pipeline STOPS if thresholds not met
    Ensures bad model never deployed!
    """

    print("="*55)
    print("CI/CD METRIC VALIDATION GATE")
    print("="*55)

    # Load metrics
    metrics_path = 'models/metrics.json'
    if not os.path.exists(metrics_path):
        print("❌ metrics.json not found!")
        print("   Run train_model.py first")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    mae  = metrics['mae']
    rmse = metrics['rmse']
    r2   = metrics['r2']
    mape = metrics['mape']

    print(f"\nLoaded metrics:")
    print(f"  Model : {metrics.get('model_name', 'RF')}")
    print(f"  MAE   : £{mae:,.2f}")
    print(f"  RMSE  : £{rmse:,.2f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.1f}%")

    # ── Define Thresholds ─────────────────────────
    # Based on our model performance
    # Set slightly below current performance
    # to allow for variation

    THRESHOLDS = {
        'max_mae' : 600,    # Max acceptable MAE
        'max_rmse': 4000,   # Max acceptable RMSE
        'min_r2'  : 0.75,   # Min acceptable R²
        'max_mape': 15.0,   # Max acceptable MAPE %
    }

    print(f"\nThresholds:")
    print(f"  MAE   < £{THRESHOLDS['max_mae']:,}")
    print(f"  RMSE  < £{THRESHOLDS['max_rmse']:,}")
    print(f"  R²    > {THRESHOLDS['min_r2']}")
    print(f"  MAPE  < {THRESHOLDS['max_mape']}%")

    # ── Run Validations ───────────────────────────
    print(f"\nValidation Results:")
    print("-"*45)

    validations = {
        'MAE check' : {
            'passed' : mae < THRESHOLDS['max_mae'],
            'value'  : f"£{mae:,.0f}",
            'threshold': f"< £{THRESHOLDS['max_mae']:,}"
        },
        'RMSE check': {
            'passed' : rmse < THRESHOLDS['max_rmse'],
            'value'  : f"£{rmse:,.0f}",
            'threshold': f"< £{THRESHOLDS['max_rmse']:,}"
        },
        'R² check'  : {
            'passed' : r2 > THRESHOLDS['min_r2'],
            'value'  : f"{r2:.4f}",
            'threshold': f"> {THRESHOLDS['min_r2']}"
        },
        'MAPE check': {
            'passed' : mape < THRESHOLDS['max_mape'],
            'value'  : f"{mape:.1f}%",
            'threshold': f"< {THRESHOLDS['max_mape']}%"
        },
    }

    all_passed = True
    for check, result in validations.items():
        status = "✅ PASS" if result['passed'] \
                 else "❌ FAIL"
        if not result['passed']:
            all_passed = False
        print(
            f"  {check:<12}: {status} "
            f"| Value={result['value']} "
            f"| Threshold={result['threshold']}"
        )

    # ── Final Decision ────────────────────────────
    print("\n" + "="*55)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("   Model approved for deployment")
        print("   Proceeding to S3 upload...")
        print("="*55)
        sys.exit(0)  # CI/CD continues ✅
    else:
        failed = [
            k for k, v in validations.items()
            if not v['passed']
        ]
        print("❌ VALIDATION FAILED!")
        print(f"   Failed checks: {failed}")
        print("   Model NOT deployed")
        print("   Fix model and retrain")
        print("="*55)
        sys.exit(1)  # CI/CD stops ❌


if __name__ == "__main__":
    validate_metrics()