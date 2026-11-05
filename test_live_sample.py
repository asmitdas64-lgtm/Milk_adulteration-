"""Live Interactive Demonstration Script for Milk Adulteration Detection.

B.Tech Minor Project — MIT Bengaluru (AY 2025–26)
Authors: Asmit Das, Suprathik Narendra, Nadia Cecil | Guide: Dr. S. Vaishnavi
"""

import sys
import pandas as pd
from foodadult.models.persist import load_model
from foodadult.predict import predict_dataframe, validate_input_data


def run_benchmark_showcase(model, features):
    """Run predefined realistic dairy laboratory scenarios."""
    scenarios = [
        {
            "name": "Scenario 1: Fresh Pure Cow Milk (Optimal Storage)",
            "data": {"ph": 6.6, "temprature": 35.0, "taste": 1, "odor": 0, "fat": 1, "turbidity": 0, "colour": 254.0},
            "expected": "Pure (Grade High)",
        },
        {
            "name": "Scenario 2: Premium Pure Milk (Room Temp)",
            "data": {"ph": 6.7, "temprature": 37.0, "taste": 1, "odor": 0, "fat": 1, "turbidity": 1, "colour": 255.0},
            "expected": "Pure (Grade High)",
        },
        {
            "name": "Scenario 3: Bacterial Souring / Curdled Milk (Lactic Acid Spikes)",
            "data": {"ph": 4.5, "temprature": 42.0, "taste": 0, "odor": 1, "fat": 0, "turbidity": 1, "colour": 245.0},
            "expected": "Adulterated / Substandard",
        },
        {
            "name": "Scenario 4: Chemically Neutralized Milk (Sodium Hydroxide / Soda)",
            "data": {"ph": 8.5, "temprature": 45.0, "taste": 0, "odor": 1, "fat": 1, "turbidity": 1, "colour": 248.0},
            "expected": "Adulterated / Substandard",
        },
        {
            "name": "Scenario 5: Watered-down Milk (Diluted Fat & Low Opacity)",
            "data": {"ph": 6.8, "temprature": 45.0, "taste": 0, "odor": 0, "fat": 0, "turbidity": 0, "colour": 240.0},
            "expected": "Adulterated / Substandard",
        },
    ]

    print("\n" + "=" * 80)
    print("LIVE DEMONSTRATION: PREDICTING 5 REALISTIC DAIRY LABORATORY SCENARIOS")
    print("=" * 80)

    for i, sc in enumerate(scenarios, 1):
        df = pd.DataFrame([sc["data"]])
        val_df = validate_input_data(df, expected_features=features)
        pred_df = predict_dataframe(model, val_df, feature_cols=features)

        pred = pred_df["prediction"].iloc[0]
        label = pred_df["predicted_label"].iloc[0]
        prob = pred_df["probability_adulterated"].iloc[0]
        conf = pred_df["confidence"].iloc[0]

        status_tag = "[PURE MILK]" if pred == 0 else "[ADULTERATED/SUBSTANDARD]"

        print(f"\n({i}) {sc['name']}")
        print(f"    Inputs: pH={sc['data']['ph']}, Temp={sc['data']['temprature']}C, Taste={sc['data']['taste']}, Odor={sc['data']['odor']}, Fat={sc['data']['fat']}, Turbidity={sc['data']['turbidity']}, Colour={sc['data']['colour']}")
        print(f"    Prediction:       {status_tag}")
        print(f"    Probability Risk: {prob:.2%} likelihood of adulteration (Confidence: {conf:.2%})")
        print(f"    Expected Status:  {sc['expected']}")
        match_str = "MATCHES EXPECTATION" if (pred == 0 and "Pure" in sc["expected"]) or (pred == 1 and "Adulterated" in sc["expected"]) else "MISMATCH"
        print(f"    Verification:     {match_str}")

    print("\n" + "=" * 80)


def interactive_test(model, features):
    """Allow user to test custom samples interactively."""
    print("\n>>> TEST YOUR OWN SAMPLE (Enter laboratory parameters):")
    try:
        ph = float(input("Enter pH (e.g. 6.6 for normal, 4.5 for sour, 8.5 for soda): "))
        temp = float(input("Enter Temperature in C (e.g. 35 to 45): "))
        taste = int(input("Enter Taste (1 for Good/Normal, 0 for Bad/Abnormal): "))
        odor = int(input("Enter Odor (0 for Normal, 1 for Foul/Pungent): "))
        fat = int(input("Enter Fat (1 for Normal/High, 0 for Low): "))
        turbidity = int(input("Enter Turbidity (0 for Normal, 1 for Abnormal): "))
        colour = float(input("Enter Colour scale (e.g. 240 to 255): "))

        sample = {
            "ph": [ph],
            "temprature": [temp],
            "taste": [taste],
            "odor": [odor],
            "fat": [fat],
            "turbidity": [turbidity],
            "colour": [colour],
        }
        df = pd.DataFrame(sample)
        val_df = validate_input_data(df, expected_features=features)
        pred_df = predict_dataframe(model, val_df, feature_cols=features)

        label = pred_df["predicted_label"].iloc[0]
        prob = pred_df["probability_adulterated"].iloc[0]
        conf = pred_df["confidence"].iloc[0]

        print("\n" + "-" * 50)
        print(f"RESULT: The sample is classified as: [{label.upper()}]")
        print(f"Adulteration Probability: {prob:.2%}")
        print(f"Model Confidence:         {conf:.2%}")
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"\n[ERROR]: Invalid input: {e}")


def main():
    model_path = "models/best_model.joblib"
    try:
        model, meta = load_model(model_path)
    except FileNotFoundError:
        print(f"[ERROR]: Model not found at {model_path}. Run 'python -m foodadult.cli evaluate' first.")
        sys.exit(1)

    features = meta.get("features", ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"])
    run_benchmark_showcase(model, features)

    # If run in an interactive terminal with arguments or stdin
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test(model, features)
    else:
        print("\n[TIP] To test custom values interactively in terminal, run:")
        print("python test_live_sample.py --interactive\n")


if __name__ == "__main__":
    main()
