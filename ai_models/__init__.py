"""
🧠 AI Models Module - Core Components

This module contains the main AI components for Amulet-AI:
- Enhanced Production System (RandomForest + Calibration + OOD)
- Two-Branch CNN System (PyTorch-based dual-view classifier)
- Compatibility Layer (for legacy model loading)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports that actually exist
try:
    from .enhanced_production_system import EnhancedProductionClassifier
    from .compatibility_loader import ProductionOODDetector, try_load_model
except ImportError as e:
    print(f"⚠️ Core AI import issue: {e}")

# Two-Branch system (optional)
try:
    from .twobranch.inference import TwoBranchInference
    from .twobranch.model import TwoBranchCNN
    from .twobranch.config import TwoBranchConfig
    _has_twobranch = True
except ImportError:
    TwoBranchInference = None
    TwoBranchCNN = None
    TwoBranchConfig = None
    _has_twobranch = False

__version__ = "4.0.0"
__author__ = "Amulet AI Team"

__all__ = [
    # Core production system
    'EnhancedProductionClassifier',
    'ProductionOODDetector',
    'try_load_model',
    
    # Two-Branch system (if available)
    'TwoBranchInference',
    'TwoBranchCNN', 
    'TwoBranchConfig'
]

# System info
def get_system_info():
    """Get AI system information"""
    return {
        'version': __version__,
        'has_enhanced_system': True,
        'has_twobranch': _has_twobranch,
        'available_components': [name for name in __all__ if globals().get(name) is not None]
    }
