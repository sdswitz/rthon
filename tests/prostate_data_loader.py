"""
Data loader for prostate cancer dataset used in C extension testing.
Converts CSV data to format suitable for C lm() function.
"""

import csv
from typing import List, Tuple, Dict

def load_prostate_data() -> Tuple[List[List[float]], List[float], List[str], Dict]:
    """
    Load prostate cancer dataset from CSV file.
    
    Returns:
        X: Design matrix (n x p) with intercept column prepended
        y: Response vector (lpsa)  
        column_names: Names of predictors including "(Intercept)"
        data_dict: Original data as dictionary for reference
    """
    import os
    
    # Get path to test data
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, '..', 'test_data', 'prostate.csv')
    
    # Read CSV data
    data_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Initialize lists for each column
        for col in reader.fieldnames:
            data_dict[col] = []
        
        # Read all rows
        for row in reader:
            for col, value in row.items():
                # Convert to float, handle missing values
                try:
                    data_dict[col].append(float(value))
                except ValueError:
                    # Handle missing or invalid values
                    data_dict[col].append(0.0)
    
    # Extract response variable (lpsa) 
    y = data_dict['lpsa']
    
    # Extract predictors in correct order
    predictor_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    
    # Build design matrix with intercept
    n = len(y)
    p = len(predictor_names) + 1  # +1 for intercept
    
    X = []
    for i in range(n):
        row = [1.0]  # Intercept term
        for pred_name in predictor_names:
            row.append(data_dict[pred_name][i])
        X.append(row)
    
    # Column names including intercept
    column_names = ['(Intercept)'] + predictor_names
    
    return X, y, column_names, data_dict

def load_reference_coefficients() -> Dict[str, Dict[str, float]]:
    """
    Load reference coefficient estimates from tidy_coefficients.csv.
    
    Returns:
        Dictionary mapping term names to their statistics
    """
    import os
    
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, '..', 'test_data', 'tidy_coefficients.csv')
    
    coeffs = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row['term']
            coeffs[term] = {
                'estimate': float(row['estimate']),
                'std_error': float(row['std.error']),  
                'statistic': float(row['statistic']),
                'p_value': float(row['p.value'])
            }
    
    return coeffs

def load_reference_model_summary() -> Dict[str, float]:
    """
    Load reference model summary statistics from model_summary.csv.
    
    Returns:
        Dictionary of model-level statistics
    """
    import os
    
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, '..', 'test_data', 'model_summary.csv')
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Should be exactly one row
        row = next(reader)
        
        return {
            'r_squared': float(row['r.squared']),
            'adj_r_squared': float(row['adj.r.squared']),
            'sigma': float(row['sigma']),
            'statistic': float(row['statistic']),  # F-statistic
            'p_value': float(row['p.value']),      # F-test p-value
            'df': int(row['df']),                  # degrees of freedom
            'df_residual': int(row['df.residual']),
            'nobs': int(row['nobs'])               # number of observations
        }

if __name__ == "__main__":
    # Test the data loader
    X, y, column_names, data_dict = load_prostate_data()
    
    print(f"Dataset shape: {len(X)} observations, {len(X[0])} variables")
    print(f"Column names: {column_names}")
    print(f"First few rows of X:")
    for i in range(3):
        print(f"  {X[i]}")
    print(f"First few y values: {y[:5]}")
    
    # Test reference data loading
    ref_coeffs = load_reference_coefficients()
    ref_summary = load_reference_model_summary()
    
    print(f"\nReference coefficients:")
    for term, stats in ref_coeffs.items():
        print(f"  {term}: {stats['estimate']:.6f}")
    
    print(f"\nReference model summary:")
    print(f"  R-squared: {ref_summary['r_squared']:.6f}")
    print(f"  Adj. R-squared: {ref_summary['adj_r_squared']:.6f}")
    print(f"  Sigma: {ref_summary['sigma']:.6f}")