
### USAGE

```
# build model
python src/model/build_model.py --data_dir output/my_video/train --config configs/my_video_config.yaml

# test model
python src/model/test_model.py output/my_video_model.pkl output/my_video_test 
```

### TODOs

* Move `extract_features` method to `Model` class
* Implement additional model types (Logistic Regression, )
* Implement other feature selection options (Chi^2, )
* Implement data transformation options (Exponential, Normalization, )
* Implement train/test data split
* Change logging from `print` statements to use `logging` module
* Change command line argument parsing to use `argparse` module
* Implement clustering for color space rather than deterministic reduction
* Add hierarchical processing to BOVW
