from setuptools import setup

setup(name='scTDA',
      version='0.97',
      description='An object oriented python library for topological data analysis of '
                  'high-throughput single-cell RNA-seq data',
      url='https://github.com/pcamara/scTDA',
      author='Pablo G. Camara',
      author_email='pablo.g.camara@gmail.com',
      license='GPL-3.0',
      packages=['scTDA'],
      install_requires=['sakmapper', 'requests', 'numexpr', 'matplotlib-venn', 'numpy', 'scipy', 'pandas',
                        'scikit-learn', 'networkx'],
      zip_safe=False)
