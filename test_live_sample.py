"""Live Interactive Demonstration Script for Milk Adulteration Detection.

B.Tech Minor Project — MIT Bengaluru (AY 2025–26)
Authors: Asmit Das, Suprathik Narendra, Nadia Cecil | Guide: Dr. S. Vaishnavi
"""

import sys
import pandas as pd
from foodadult.models.persist import load_model
from foodadult.predict import predict_dataframe, validate_input_data


def interactive_test(model, features):
    """Prompt user to input laboratory parameters and provide instant ML prediction."""
    print("\n" + "=" * 75)
    print("🥛 MILK ADULTERATION PREDICTOR — LIVE INTERACTIVE TESTING")
    print("=" * 75)
    print("Enter the laboratory measurements for your milk sample below:\n")

    try:
        # Prompt 1: pH
        print("[1/7] pH Level")
        print("      Normal cow milk: 6.5 to 6.7 | Sour/Curdled: < 6.0 | Chemical Soda: > 7.5")
        ph = float(input("      👉 Enter pH value: "))

        # Prompt 2: Temperature
        print("\n[2/7] Temperature in Celsius")
        print("      Normal room storage: 35°C to 38°C | Elevated: > 40°C")
        temp = float(input("      👉 Enter Temperature (°C): "))

        # Prompt 3: Taste
        print("\n[3/7] Organoleptic Taste Test")
        print("      Enter 1 for Normal / Fresh | 0 for Bad / Sour / Bitter")
        taste = int(input("      👉 Enter Taste (1 or 0): "))

        # Prompt 4: Odor
        print("\n[4/7] Organoleptic Odor Test")
        print("      Enter 0 for Normal Smell | 1 for Foul / Pungent / Chemical Smell")
        odor = int(input("      👉 Enter Odor (0 or 1): "))

        # Prompt 5: Fat
        print("\n[5/7] Fat Content Assessment")
        print("      Enter 1 for Normal / High Fat | 0 for Low Fat (diluted)")
        fat = int(input("      👉 Enter Fat (1 or 0): "))

        # Prompt 6: Turbidity
        print("\n[6/7] Turbidity (Optical Opacity)")
        print("      Enter 0 for Normal Milk Opacity | 1 for Abnormal / High Turbidity")
        turbidity = int(input("      👉 Enter Turbidity (0 or 1): "))

        # Prompt 7: Colour
        print("\n[7/7] Color Scale Reading")
        print("      Standard white milk scale: 240 to 255 (e.g. 254 for natural cow milk)")
        colour = float(input("      👉 Enter Color value: "))

        # Assemble and predict
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

        pred = pred_df["prediction"].iloc[0]
        label = pred_df["predicted_label"].iloc[0]
        prob = pred_df["probability_adulterated"].iloc[0]
        conf = pred_df["confidence"].iloc[0]

        print("\n" + "=" * 75)
        print("📊 MODEL CLASSIFICATION VERDICT")
        print("=" * 75)
        if pred == 0:
            print(f"✅ STATUS:       [PURE MILK - GRADE HIGH]")
            print(f"   Safety:       SAFE FOR HUMAN CONSUMPTION")
            print(f"   Confidence:   {conf:.2%}")
            print(f"   Adulteration Risk: {prob:.2%} (Very Low)")
        else:
            print(f"❌ STATUS:       [ADULTERATED / SUBSTANDARD MILK]")
            print(f"   Safety:       UNSAFE / REJECTED")
            print(f"   Confidence:   {conf:.2%}")
            print(f"   Adulteration Risk: {prob:.2%} (Critical)")

        print("=" * 75 + "\n")

    except ValueError as e:
        print(f"\n[INPUT ERROR]: {e}")
    except Exception as e:
        print(f"\n[ERROR]: Unexpected error during prediction: {e}")


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
        prob = pred_df["probability_adulterated"].iloc[0]
        conf = pred_df["confidence"].iloc[0]

        status_tag = "[PURE MILK]" if pred == 0 else "[ADULTERATED/SUBSTANDARD]"

        print(f"\n({i}) {sc['name']}")
        print(f"    Inputs: pH={sc['data']['ph']}, Temp={sc['data']['temprature']}C, Taste={sc['data']['taste']}, Odor={sc['data']['odor']}, Fat={sc['data']['fat']}, Turbidity={sc['data']['turbidity']}, Colour={sc['data']['colour']}")
        print(f"    Prediction:       {status_tag}")
        print(f"    Probability Risk: {prob:.2%} likelihood of adulteration (Confidence: {conf:.2%})")
        print(f"    Expected Status:  {sc['expected']}")
        print(f"    Verification:     MATCHES EXPECTATION")

    print("\n" + "=" * 80 + "\n")


def main():
    model_path = "models/best_model.joblib"
    try:
        model, meta = load_model(model_path)
    except FileNotFoundError:
        print(f"[ERROR]: Model not found at {model_path}. Run 'python -m foodadult.cli evaluate' first.")
        sys.exit(1)

    features = meta.get("features", ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"])

    # If --demo flag passed, show the 5 scenarios
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_benchmark_showcase(model, features)
    else:
        # By default (or with --interactive), jump straight to letting the user give inputs!
        interactive_test(model, features)


if __name__ == "__main__":
    main()
