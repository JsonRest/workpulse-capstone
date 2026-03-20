"""
Test WorkPulse Vertex AI Endpoint
=================================
Usage:
    pip install google-cloud-aiplatform
    python test_vertex_endpoint.py --project YOUR_PROJECT_ID --endpoint YOUR_ENDPOINT_ID
"""

import argparse
from google.cloud import aiplatform


def test_endpoint(project_id: str, region: str, endpoint_id: str):
    """Send test predictions to the deployed Vertex AI endpoint."""

    aiplatform.init(project=project_id, location=region)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    # Feature order:
    # [overtime_index, wellbeing_composite, workload_pressure, satisfaction_gap,
    #  high_stress_flag, tenure_risk_flag, job_satisfaction, work_life_balance,
    #  log_income, monthly_income, tenure_years, age, age_group]

    print("=" * 60)
    print("WorkPulse Vertex AI Endpoint Test")
    print("=" * 60)

    # Test 1: High-risk employee
    print("\nTest 1: HIGH RISK employee")
    high_risk = [0.75, 0.25, 0.7, -0.3, 1, 1, 0.3, 0.2, 7.5, 3000, 2, 27, 0]
    result = endpoint.predict(instances=[high_risk])
    pred = result.predictions[0]
    print(f"  Risk: {pred['burnout_risk']} ({pred['risk_level']})")
    print(f"  Probability: {pred['burnout_probability']}")
    if pred.get("top_factors"):
        print(f"  Top factors:")
        for f in pred["top_factors"][:3]:
            print(f"    {f['feature']}: {f['importance']}")

    # Test 2: Low-risk employee
    print("\nTest 2: LOW RISK employee")
    low_risk = [0.1, 0.85, 0.1, 0.2, 0, 0, 0.8, 0.9, 9.5, 12000, 8, 42, 2]
    result = endpoint.predict(instances=[low_risk])
    pred = result.predictions[0]
    print(f"  Risk: {pred['burnout_risk']} ({pred['risk_level']})")
    print(f"  Probability: {pred['burnout_probability']}")

    # Test 3: Batch prediction (3 employees)
    print("\nTest 3: BATCH prediction (3 employees)")
    batch = [
        high_risk,
        low_risk,
        [0.4, 0.5, 0.35, -0.1, 1, 0, 0.5, 0.5, 8.0, 5000, 3, 30, 1],  # medium risk
    ]
    result = endpoint.predict(instances=batch)
    for i, pred in enumerate(result.predictions):
        print(f"  Employee {i+1}: {pred['risk_level']} (P={pred['burnout_probability']})")

    print("\n" + "=" * 60)
    print("All tests complete ✅")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test WorkPulse Vertex AI endpoint")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--endpoint", required=True, help="Vertex AI endpoint ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    args = parser.parse_args()

    test_endpoint(args.project, args.region, args.endpoint)


if __name__ == "__main__":
    main()
