
### Usage
```
>>> python src/model/build_model.py -h

usage: python src/model/build_model.py [-h] [-v] [-e ERROR_INFO_FP]
                                       [-store-all]
                                       [-visualize VISUALIZE_OPTIONS]
                                       [-save-vis] -train TRAIN_DIR -test
                                       TEST_DIR -c CONFIG -m MODEL_OUTPUT_FP

Build an image classification model.

required arguments:
  -train TRAIN_DIR      The location of the training snippets
  -test TEST_DIR        The location of the testing snippets
  -c CONFIG             The location of the config file with the model
                        parameters
  -m MODEL_OUTPUT_FP    The output location of the created model

optional arguments:
  -h, --help            show this help message and exit
  -v                    Verbose output (debug)
  -e ERROR_INFO_FP      The output location of cross validation errors for
                        each config, for analysis after training is complete
  -store-all            Store all models (default: store only the best models)
  -visualize VISUALIZE_OPTIONS
                        Create visualizations of model components and/or
                        parameters
  -save-vis             Save visualizations to file
```

### Example

```
python src/model/build_model.py -train output/my_example_train/ -test output/my_example_test -c configs/my_example/config.yaml -m models/my_example_model.pkl -visualize bovw_pca -save-vis
```

### Details

Model architecture is as follows:
* Image descriptors of image keypoints are aggregated from image snippets created in Step 1 (data preparation) using OpenCV descriptor extractors (KAZE, ORB, etc). Image snippets can be blurred for (usually) better performance.
* Set of image descriptors is clustered (kmeans, hierarchical clustering) to form Bag of Visual Words (BOVW). Number of clusters can be specified, otherwise many different cluster sizes are tried and the best number is used.
* Histograms of BOVW and color info are created from training image snippets.
* Histograms are used as input to model of choice (SVM, KNN, Logistic Regression). Before training/testing, histograms are transformed using the some combination of data transformation (scaling, normalization), feature selection (variance threshold, Chi^2), and approximation kernel mapping (RBF, Chi^2).
* Model supports hyperparameter search where a list of potential parameters are considered and the best is chosen using k-folds cross validation.
* Various model details can be visualized, such as BOVW structure and how the performance varies with different parameters.


### TODOs

* Implement other feature selection options (Chi^2, )
* Implement data transformation options (Exponential, Normalization, )

