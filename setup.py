from setuptools import setup



setup(
    name='paper-generalizability-window-size',
    version='1.0',
    description='scripts',
    license="Concordia University",
    
    author='',
    author_email='',
    REQUIRES_PYTHON='>=3.0.0',

    REQUIRED=['numpy', 'pandas', 'sklearn', 'matplotlib'],  # external packages as dependencies
    scripts=[
        'Scripts/Classification.py',
        'Scripts/plot_figures.py',
        'Scripts/plot_results.py',
        'Scripts/Preprocessing.py'
    ]
)
