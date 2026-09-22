from pathlib import Path
import joblib
import mlflow

def register_existing_artifacts():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Qafzah_Inference_Pipeline")

    artifacts_dir = Path("data/artifacts")
    model_path = artifacts_dir / "final_model.joblib"

    with mlflow.start_run(run_name="register_baseline_artifacts") as run:
        if artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")

        if model_path.exists():
            model = joblib.load(model_path)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="LateDeliveryModel",
                serialization_format="cloudpickle"
            )
            print(f"Model successfully registered under Run ID: {run.info.run_id}")

if __name__ == "__main__":
    register_existing_artifacts()
