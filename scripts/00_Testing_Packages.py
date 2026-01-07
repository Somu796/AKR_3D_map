def test_imports():
    """Test package imports."""
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'xarray': 'Xarray',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'netCDF4': 'NetCDF4',
        'plotly': 'Plotly',
    }
    
    print("="*60)
    print("CHECKING INSTALLED PACKAGES")
    print("="*60)
    
    all_good = True
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f" {package_name:20s} - OK")
        except ImportError as e:
            print(f" {package_name:20s} - MISSING")
            all_good = False
    
    if all_good:
        print("\n" + "="*60)
        print(" All packages installed successfully!")
        print("="*60)
        
        # Quick functionality test
        import numpy as np
        import pandas as pd
        import xarray as xr
        
        print("\nQuick test:")
        ds = xr.Dataset({
            'test': (['x', 'y'], np.random.rand(3, 3))
        })
        print(" Created test xarray Dataset")
        print(ds)
        print("\n Ready to start your AKR analysis!")
    else:
        print("\n Some packages missing - run:")
        print("   uv add xarray numpy pandas matplotlib scipy netcdf4")
    
    return all_good

if __name__ == '__main__':
    test_imports()