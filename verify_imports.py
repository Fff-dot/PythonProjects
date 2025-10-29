modules = [
    'numpy',
    'pandas',
    'datetime',
    'matplotlib',
    'matplotlib.pyplot',
    'seaborn',
    'sklearn.preprocessing',
    'sklearn',
    'yellowbrick',
]

if __name__ == '__main__':
    import importlib, sys
    for m in modules:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"OK: {m} (version: {ver})")
        except Exception as e:
            print(f"FAIL: {m} -> {e}")
    print('Python executable:', sys.executable)
