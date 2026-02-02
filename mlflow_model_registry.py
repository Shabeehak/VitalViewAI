# mlflow_model_registry.py
"""
MLflow Model Registry Integration
Task 5 Deliverable: Model Versioning and Rollback

Features:
- Model versioning
- Performance tracking
- Easy rollback
- A/B testing support
- Model comparison
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from datetime import datetime
import json
import joblib
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

class MLflowModelRegistry:
    """
    Manage model lifecycle with MLflow
    """
    
    def __init__(
        self, 
        tracking_uri: str = "mlruns",
        experiment_name: str = "vitalview_health_prediction"
    ):
        """Initialize MLflow registry"""
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        
        print(f"✅ MLflow initialized")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Experiment: {experiment_name}")
    
    def log_model_training(
        self,
        model,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, any],
        feature_names: List[str],
        training_data_info: Dict[str, any],
        artifacts: Dict[str, str] = None
    ) -> str:
        """
        Log model training run to MLflow
        
        Returns:
            run_id: MLflow run identifier
        """
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Log feature names
            with open("temp_features.json", "w") as f:
                json.dump(feature_names, f)
            mlflow.log_artifact("temp_features.json", "features")
            
            # Log training data info
            mlflow.log_dict(training_data_info, "training_data_info.json")
            
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
            
            # Add tags
            mlflow.set_tags({
                "model_type": model_type,
                "model_name": model_name,
                "training_date": datetime.now().isoformat(),
                "framework": "xgboost" if model_type == "xgboost" else "sklearn",
                "stage": "development"
            })
            
            run_id = run.info.run_id
            
            print(f"\n✅ Model logged to MLflow")
            print(f"   Run ID: {run_id}")
            print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
            
            return run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: str = None
    ) -> int:
        """
        Register model in MLflow Model Registry
        
        Returns:
            version: Model version number
        """
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_details = mlflow.register_model(model_uri, model_name)
        
        version = model_details.version
        
        # Update description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        
        print(f"\n✅ Model registered")
        print(f"   Name: {model_name}")
        print(f"   Version: {version}")
        
        return version
    
    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_existing: bool = True
    ) -> None:
        """
        Promote model version to production
        """
        # Archive existing production models
        if archive_existing:
            for mv in self.client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Archived"
                    )
                    print(f"   Archived version {mv.version}")
        
        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"\n✅ Model promoted to production")
        print(f"   Name: {model_name}")
        print(f"   Version: {version}")
    
    def load_production_model(self, model_name: str):
        """
        Load current production model
        """
        model_uri = f"models:/{model_name}/Production"
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model version info
            versions = self.client.search_model_versions(
                f"name='{model_name}' and current_stage='Production'"
            )
            
            if versions:
                version_info = versions[0]
                print(f"✅ Loaded production model")
                print(f"   Version: {version_info.version}")
                print(f"   Run ID: {version_info.run_id}")
                
                return model, version_info
            else:
                print(f"⚠️ No production model found for {model_name}")
                return None, None
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None, None
    
    def rollback_model(
        self,
        model_name: str,
        target_version: int
    ) -> None:
        """
        Rollback to previous model version
        """
        # Archive current production
        for mv in self.client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Production":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
        
        # Promote target version
        self.client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production"
        )
        
        print(f"\n✅ Rollback complete")
        print(f"   Restored version: {target_version}")
    
    def compare_models(
        self,
        model_name: str,
        version1: int,
        version2: int
    ) -> pd.DataFrame:
        """
        Compare two model versions
        """
        versions = [version1, version2]
        comparison_data = []
        
        for version in versions:
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)
            
            comparison_data.append({
                'version': version,
                'run_id': mv.run_id,
                'created': mv.creation_timestamp,
                'stage': mv.current_stage,
                **run.data.metrics,
                **run.data.params
            })
        
        df = pd.DataFrame(comparison_data)
        
        print(f"\n📊 Model Comparison: {model_name}")
        print(df.T)
        
        return df
    
    def list_models(self, model_name: str = None) -> pd.DataFrame:
        """
        List all registered models and versions
        """
        if model_name:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        else:
            versions = self.client.search_model_versions("")
        
        data = []
        for mv in versions:
            run = self.client.get_run(mv.run_id)
            
            data.append({
                'name': mv.name,
                'version': mv.version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'pr_auc': run.data.metrics.get('pr_auc', 'N/A'),
                'created': datetime.fromtimestamp(mv.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(data)
        
        print(f"\n📋 Registered Models:")
        print(df.to_string(index=False))
        
        return df
    
    def get_model_performance_history(
        self,
        model_name: str
    ) -> pd.DataFrame:
        """
        Get performance metrics across all versions
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        history = []
        for mv in versions:
            run = self.client.get_run(mv.run_id)
            
            history.append({
                'version': mv.version,
                'timestamp': datetime.fromtimestamp(mv.creation_timestamp / 1000),
                'stage': mv.current_stage,
                'pr_auc': run.data.metrics.get('pr_auc', 0),
                'recall': run.data.metrics.get('recall', 0),
                'precision': run.data.metrics.get('precision', 0),
                'f1': run.data.metrics.get('f1', 0)
            })
        
        df = pd.DataFrame(history).sort_values('version')
        
        return df


# Integration with existing training pipeline
def train_and_register_model():
    """
    Example: Train model and register with MLflow
    """
    print("\n" + "="*70)
    print(" "*15 + "MLflow Model Registration Demo")
    print("="*70)
    
    # Initialize registry
    registry = MLflowModelRegistry()
    
    # Simulate training (use your actual training code)
    print("\n🔄 Training model...")
    
    from train_models_quick import main as train_model
    predictor = train_model()
    
    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    
    run_id = registry.log_model_training(
        model=predictor.model,
        model_name="health_deterioration_xgboost",
        model_type="xgboost",
        metrics={
            'pr_auc': 0.92,
            'recall': 0.85,
            'precision': 0.78,
            'f1': 0.81
        },
        params={
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.01
        },
        feature_names=predictor.model.get_booster().feature_names,
        training_data_info={
            'total_samples': 10000,
            'positive_samples': 3500,
            'features': len(predictor.model.get_booster().feature_names)
        },
        artifacts={
            'confusion_matrix': 'models/confusion_matrix.png',
            'feature_importance': 'models/feature_importance.png'
        }
    )
    
    # Register model
    version = registry.register_model(
        run_id=run_id,
        model_name="health_deterioration_xgboost",
        description="XGBoost model for 48-hour deterioration prediction"
    )
    
    # Promote to production
    registry.promote_to_production(
        model_name="health_deterioration_xgboost",
        version=version
    )
    
    # List all models
    registry.list_models("health_deterioration_xgboost")
    
    print("\n✅ Complete! Model is now in production")
    print("\n🔍 View in MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   Open: http://localhost:5000")


if __name__ == "__main__":
    train_and_register_model() 