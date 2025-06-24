from functools import wraps
import numpy as np

def patch_get_feature_names_out(estimator, prefix=None):
    """
    Monkey-patch estimator to preserve input names or use custom prefix for output feature names.
    
    Args:
        estimator: The scikit-learn estimator to modify
        prefix: Custom prefix for feature names. If None, preserves input names when possible.
    """
    original_get_feature_names_out = estimator.get_feature_names_out
    
    @wraps(original_get_feature_names_out)
    def custom_get_feature_names_out(input_features=None):
        original_names = original_get_feature_names_out(input_features)
        
        if prefix is None:
            feature_names_in = getattr(estimator, 'feature_names_in_', None)
            if feature_names_in is not None:
                # Truncate if input features are longer than output
                if len(feature_names_in) >= len(original_names):
                    return feature_names_in[:len(original_names)]
            return original_names
        
        # Use custom prefix
        return np.array([f"{prefix}{i}" for i in range(len(original_names))], dtype=object)
    
    estimator.get_feature_names_out = custom_get_feature_names_out
    return estimator