"""
Model Registry with Versioning and Rollback
Task 5: Scalability - Model version management
"""

import json
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import joblib


class ModelRegistry:
    """
    Simple model registry for version management and rollback
    
    Demonstrates:
    - Model versioning
    - Rollback capability
    - Metadata tracking
    - Production/staging separation
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'production': None,
            'staging': None
        }
    
    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        model_type: str,
        metrics: Dict,
        description: str = ""
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_path: Path to trained model file
            model_type: 'xgboost', 'lstm', etc.
            metrics: Performance metrics (pr_auc, recall, etc.)
            description: Version description
            
        Returns:
            version_id: Unique version identifier
        """
        # Create version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_type}_v{timestamp}"
        
        # Create version directory
        version_dir = self.registry_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_filename = Path(model_path).name
        target_path = version_dir / model_filename
        shutil.copy2(model_path, target_path)
        
        # Copy metadata if exists
        metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
        if metadata_path.exists():
            shutil.copy2(metadata_path, version_dir / "metadata.json")
        
        # Register in metadata
        self.metadata['models'][version_id] = {
            'version_id': version_id,
            'model_type': model_type,
            'model_path': str(target_path),
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'description': description,
            'status': 'registered'
        }
        
        self._save_metadata()
        
        print(f"✅ Registered model version: {version_id}")
        print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        print(f"   Path: {target_path}")
        
        return version_id
    
    def promote_to_staging(self, version_id: str):
        """Promote model to staging environment"""
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version {version_id} not found")
        
        self.metadata['staging'] = version_id
        self.metadata['models'][version_id]['status'] = 'staging'
        self._save_metadata()
        
        print(f"✅ Promoted {version_id} to STAGING")
    
    def promote_to_production(self, version_id: str):
        """
        Promote model to production
        
        Creates rollback point automatically
        """
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version {version_id} not found")
        
        # Store previous production as rollback
        if self.metadata['production']:
            prev_version = self.metadata['production']
            self.metadata['models'][prev_version]['status'] = 'rollback_available'
            print(f"📦 Previous production ({prev_version}) available for rollback")
        
        # Promote new version
        self.metadata['production'] = version_id
        self.metadata['models'][version_id]['status'] = 'production'
        self._save_metadata()
        
        print(f"✅ Promoted {version_id} to PRODUCTION")
        
        # Copy to production location
        self._deploy_to_production(version_id)
    
    def _deploy_to_production(self, version_id: str):
        """Copy model to production location"""
        model_info = self.metadata['models'][version_id]
        source_path = Path(model_info['model_path'])
        
        # Production paths
        prod_dir = Path("models")
        prod_model_path = prod_dir / "xgboost_model.pkl"
        prod_metadata_path = prod_dir / "xgboost_model_metadata.json"
        
        # Copy model
        shutil.copy2(source_path, prod_model_path)
        
        # Copy metadata
        version_dir = source_path.parent
        if (version_dir / "metadata.json").exists():
            shutil.copy2(version_dir / "metadata.json", prod_metadata_path)
        
        print(f"   Deployed to: {prod_model_path}")
    
    def rollback(self):
        """
        Rollback to previous production model
        
        Returns:
            version_id: Version that was rolled back to
        """
        # Find rollback version
        rollback_version = None
        for version_id, info in self.metadata['models'].items():
            if info['status'] == 'rollback_available':
                rollback_version = version_id
                break
        
        if not rollback_version:
            raise ValueError("No rollback version available")
        
        print(f"⚠️  Rolling back to: {rollback_version}")
        
        # Demote current production
        current_prod = self.metadata['production']
        if current_prod:
            self.metadata['models'][current_prod]['status'] = 'archived'
        
        # Promote rollback version
        self.promote_to_production(rollback_version)
        
        return rollback_version
    
    def list_models(self, status: Optional[str] = None) -> List[Dict]:
        """List all registered models"""
        models = []
        for version_id, info in self.metadata['models'].items():
            if status is None or info['status'] == status:
                models.append(info)
        
        # Sort by registration time (newest first)
        models.sort(key=lambda x: x['registered_at'], reverse=True)
        return models
    
    def get_production_model(self) -> Optional[Dict]:
        """Get currently deployed production model info"""
        prod_version = self.metadata['production']
        if prod_version:
            return self.metadata['models'][prod_version]
        return None
    
    def compare_models(self, version_id_1: str, version_id_2: str):
        """Compare metrics between two model versions"""
        if version_id_1 not in self.metadata['models']:
            raise ValueError(f"Model {version_id_1} not found")
        if version_id_2 not in self.metadata['models']:
            raise ValueError(f"Model {version_id_2} not found")
        
        model_1 = self.metadata['models'][version_id_1]
        model_2 = self.metadata['models'][version_id_2]
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        print(f"Version 1: {version_id_1}")
        print(f"  Status: {model_1['status']}")
        print(f"  Metrics: {model_1['metrics']}")
        print()
        
        print(f"Version 2: {version_id_2}")
        print(f"  Status: {model_2['status']}")
        print(f"  Metrics: {model_2['metrics']}")
        print()
        
        # Compare PR-AUC
        pr_auc_1 = model_1['metrics'].get('pr_auc', 0)
        pr_auc_2 = model_2['metrics'].get('pr_auc', 0)
        
        if pr_auc_1 > pr_auc_2:
            print(f"✅ {version_id_1} performs better (+{pr_auc_1 - pr_auc_2:.4f})")
        else:
            print(f"✅ {version_id_2} performs better (+{pr_auc_2 - pr_auc_1:.4f})")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Demo of model registry"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Registry CLI')
    parser.add_argument('command', choices=['register', 'promote', 'rollback', 'list', 'compare'])
    parser.add_argument('--model-path', help='Path to model file')
    parser.add_argument('--version', help='Model version ID')
    parser.add_argument('--version2', help='Second model version for comparison')
    parser.add_argument('--type', default='xgboost', help='Model type')
    parser.add_argument('--pr-auc', type=float, help='PR-AUC metric')
    parser.add_argument('--description', default='', help='Version description')
    
    args = parser.parse_args()
    
    registry = ModelRegistry()
    
    if args.command == 'register':
        if not args.model_path or args.pr_auc is None:
            print("Error: --model-path and --pr-auc required")
            return
        
        version_id = registry.register_model(
            model_path=args.model_path,
            model_type=args.type,
            metrics={'pr_auc': args.pr_auc},
            description=args.description
        )
        
        print(f"\nNext steps:")
        print(f"  1. Test: python model_registry.py promote --version {version_id}")
        print(f"  2. Deploy: Promote to production when ready")
    
    elif args.command == 'promote':
        if not args.version:
            print("Error: --version required")
            return
        
        # Promote to staging first
        registry.promote_to_staging(args.version)
        
        # Ask for confirmation for production
        confirm = input("Promote to PRODUCTION? (yes/no): ")
        if confirm.lower() == 'yes':
            registry.promote_to_production(args.version)
    
    elif args.command == 'rollback':
        confirm = input("⚠️  ROLLBACK to previous version? (yes/no): ")
        if confirm.lower() == 'yes':
            registry.rollback()
    
    elif args.command == 'list':
        models = registry.list_models()
        
        print(f"\n{'='*80}")
        print(f"REGISTERED MODELS ({len(models)} total)")
        print(f"{'='*80}\n")
        
        for model in models:
            status_emoji = {
                'production': '🟢',
                'staging': '🟡',
                'rollback_available': '📦',
                'registered': '⚪',
                'archived': '⚫'
            }
            
            emoji = status_emoji.get(model['status'], '⚪')
            print(f"{emoji} {model['version_id']}")
            print(f"   Status: {model['status']}")
            print(f"   PR-AUC: {model['metrics'].get('pr_auc', 'N/A'):.4f}")
            print(f"   Registered: {model['registered_at']}")
            print()
    
    elif args.command == 'compare':
        if not args.version or not args.version2:
            print("Error: --version and --version2 required")
            return
        
        registry.compare_models(args.version, args.version2)


if __name__ == "__main__":
    main()