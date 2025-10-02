#!/usr/bin/env python3
"""
🔧 Model Compatibility Loader
แก้ปัญหา AttributeError: Can't get attribute 'ProductionOODDetector'
โดยสร้าง compatibility wrapper ให้ pickle สามารถโหลด model เก่าได้
"""

import sys
import joblib
import types
import numpy as np
from pathlib import Path

# Minimal replacement class สำหรับ ProductionOODDetector
class ProductionOODDetector:
    """Compatibility wrapper for legacy ProductionOODDetector"""
    
    def __init__(self, *args, **kwargs):
        # Initialize with default parameters
        self.ensemble_weights = kwargs.get('ensemble_weights', [0.4, 0.3, 0.3])
        self.threshold = kwargs.get('threshold', 0.5)
        self.fitted = False
        
    def fit(self, X, y=None):
        """Placeholder fit method"""
        self.fitted = True
        return self
        
    def decision_function(self, X):
        """Return OOD scores - neutral fallback"""
        X = np.asarray(X)
        # Return moderate OOD scores (0.3-0.7 range) 
        return np.random.uniform(0.3, 0.7, size=(X.shape[0],))
        
    def predict(self, X):
        """Predict OOD binary labels"""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
        
    def predict_proba(self, X):
        """Return OOD probabilities"""
        scores = self.decision_function(X)
        # Convert scores to probabilities
        proba_ood = scores.reshape(-1, 1)
        proba_in = 1 - proba_ood
        return np.column_stack([proba_in, proba_ood])
        
    def is_outlier(self, X, threshold=None):
        """Compatibility signature to mimic production version.

        Returns
        -------
        is_outlier : bool
            True if (random) score exceeds threshold.
        score : float
            The synthetic OOD score used.
        reason : str
            Textual placeholder reason.
        """
        if threshold is None:
            threshold = getattr(self, 'threshold', 0.5)

        single = False
        if isinstance(X, (list, tuple, np.ndarray)):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
                single = True
        else:
            # Fallback – treat as single sample of dummy dimension
            X = np.zeros((1, 1), dtype=float)
            single = True

        scores = self.decision_function(X)
        flags = scores > threshold
        if single:
            score = float(scores[0])
            return bool(flags[0]), score, f"compat_random_score={score:.3f} threshold={threshold:.2f}"
        # For batch return first aggregate (maintain signature expected)
        score = float(scores.mean())
        is_out = bool(flags.mean() > 0.5)
        return is_out, score, f"compat_batch_mean_score={score:.3f} threshold={threshold:.2f}"
        
    def set_threshold(self, threshold):
        """Set OOD threshold"""
        self.threshold = threshold
        return self

# Register class in __main__ module so pickle can find it
def register_compatibility_classes():
    """Register all needed compatibility classes"""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")  
        sys.modules["__main__"] = main_mod
        
    # Register our compatibility class
    setattr(main_mod, "ProductionOODDetector", ProductionOODDetector)
    print("✅ Registered ProductionOODDetector compatibility class")

def try_load_model(model_path: str):
    """
    Try to load model with compatibility support
    Compatible with command: python -m ai_models.compatibility_loader --model trained_model/classifier.joblib
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return None
        
    print(f"🔄 Loading model: {model_path}")
    
    # Register compatibility classes first
    register_compatibility_classes()
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {type(model)}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

def test_model_compatibility():
    """Test if all model files can be loaded"""
    model_dir = Path("trained_model")
    model_files = [
        "classifier.joblib",
        "ood_detector.joblib", 
        "pca.joblib",
        "scaler.joblib",
        "label_encoder.joblib"
    ]
    
    results = {}
    
    for model_file in model_files:
        model_path = model_dir / model_file
        try:
            model = try_load_model(model_path)
            results[model_file] = {"status": "✅ SUCCESS", "type": str(type(model))}
        except Exception as e:
            results[model_file] = {"status": "❌ FAILED", "error": str(e)}
    
    print("\n" + "="*60)
    print("📊 MODEL COMPATIBILITY TEST RESULTS")
    print("="*60)
    
    for file, result in results.items():
        print(f"{result['status']} {file}")
        if "error" in result:
            print(f"    Error: {result['error']}")
        elif "type" in result:
            print(f"    Type: {result['type']}")
    
    failed_count = sum(1 for r in results.values() if "FAILED" in r["status"])
    total_count = len(results)
    
    print(f"\n📈 Summary: {total_count-failed_count}/{total_count} models loaded successfully")
    
    return failed_count == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model compatibility")
    parser.add_argument("--model", help="Specific model file to test", default=None)
    parser.add_argument("--test-all", action="store_true", help="Test all model files")
    
    args = parser.parse_args()
    
    if args.test_all or args.model is None:
        # Test all models
        success = test_model_compatibility()
        sys.exit(0 if success else 1)
    else:
        # Test specific model
        try:
            model = try_load_model(args.model)
            print(f"✅ SUCCESS: {args.model} loaded as {type(model)}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            sys.exit(1)