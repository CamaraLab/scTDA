# scTDA
scTDA is an object oriented python library for topological data analysis of high-throughput single-cell RNA-seq
data. It includes tools for the preprocessing, analysis, and exploration of single-cell RNA-seq data based on topological representations.

## Installation

To install scTDA run:

`pip install scTDA`

Alternatively, to install the most updated version you can download the source code and run:

`python setup.py install`

For optimal visualization results it is strongly recommended to have Graphviz tools and PyGraphviz installed.

## Docker

A Docker container with a fully configured jupyter notebook envirnoment, graphviz, and scTDA can be obtained running:

`docker pull pcamara/sctda`

To run the container use:

`docker run -it -v /path/to/your/working/directory:/home/jovyan/work --rm -p 8888:8888 pcamara/sctda`

where `/path/to/your/working/directory` is the folder containing the data you want to analyze. In some systems it may be required replacing `/home/jovyan/work` with `//home/jovyan/work` in the above command.

## Using scTDA

scTDA can be imported using the command:

`import scTDA`

A tutorial illustrating the basic scTDA workflow can be found in `doc/scTDA Tutorial.html`. The source notebook and data files for the 
tutorial can be downloaded [here](https://www.dropbox.com/s/ma80a641miteyxf/scTDA%20Tutorial.tar.gz?dl=0). For optimal visualization when working with notebooks, we recommend using `%matplotlib notebook`.

More details on the scTDA algorithm can be found in:

Rizvi, A. H.\*, Camara, P. G.\*, Kandror, E. K., Roberts, T. J., Scheiren, I., Maniatis, T., and Rabadan, R., 
"Single-Cell Topological RNA-Seq Analysis Reveals Insights Into Cellular Differentiation and Development", Nat. Biotechnol. (2017) 35: 551-560.
[\* These authors contributed equally to this work.]
