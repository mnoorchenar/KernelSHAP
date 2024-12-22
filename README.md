# mean_kernel_shap_with_constraint

The `mean_kernel_shap_with_constraint` function computes SHAP values using a kernel-based method while enforcing the additive efficiency constraint. It provides an efficient and robust approach for model-agnostic feature attribution.

## Function Description

```python
def mean_kernel_shap_with_constraint(f, x, reference, M, nsamples="auto"):
    """
    Kernel SHAP with additive efficiency constraint and kernel weights.
    
    Parameters:
    - f: The model function to explain.
    - x: Instance to explain (1D array of feature values).
    - reference: Reference value for each feature (1D array, usually the mean of the dataset).
    - M: Number of features.
    - nsamples: Number of samples for the sampling method. If "auto", uses nsamples = min(2 * M + 2048, 2^M).
    
    Returns:
    - shap_values: Shapley values (1D array of size M).
    - baseline: The baseline value (φ₀).
    """
