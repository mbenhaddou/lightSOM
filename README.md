lightSOM
-----
A Python Library for Self Organizing Map (SOM)

As much as possible, the structure of SOM is similar to `somtoolbox` in Matlab. It has the following functionalities:

1. Only Batch training, which is faster than online training. It has parallel processing option similar to `sklearn` format and it speeds up the training procedure, but it depends on the data size and mainly the size of the SOM grid.I couldn't manage the memory problem and therefore, I recommend single core processing at the moment. But nevertheless, the implementation of the algorithm is carefully done for all those important matrix calculations, such as `scipy` sparse matrix and `numexpr` for calculation of Euclidean distance.
2. PCA (or RandomPCA (default)) initialization, using `sklearn` or random initialization.
3. component plane visualization (different modes).
4. Hitmap.
5. U-Matrix visualization.
6. 1-d or 2-d SOM with only rectangular, planar grid. (works well in comparison with hexagonal shape, when I was checking in Matlab with somtoolbox).
7. Different methods for function approximation and predictions (mostly using Sklearn).


Quality Measures
----------------

After the SOM has been trained, the map needs to be evaluated to find out if it has been optimally trained, or if further training is required. The SOM quality is usually measured with two criteria: quantization error (QE) and topographic error (TE). The QE is the average distance between each data point and its BMU, and TE represents the proportion of all data for which the first and second BMU are not adjacent with respect to the measurement of topology preservation (Kohonen, 2001).

### Dependencies:
SOMPY has the following dependencies:
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas


### Installation:
```Python
pip install lightSOM
```

