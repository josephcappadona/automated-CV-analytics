
## Steps

### Step 1: Data Preparation

Responsible for extracting and creating the training data from the video/image source.

`extract_frames.py` uses `ffmpeg` to extract frames from a source video and writes them to a frames directory.

`apply_label_template.py` takes a static label template and applies it to all of the images in the specified folder. This is useful when an entity appears in the same location across all video frames.

`label_template.py` contains the library functions that `apply_label_template.py` uses to load and edit label templates.

`create_snippets.py` extracts the labeled subimages and writes them to a snippets directory.


### Step 2: Model Creation

Implements image classification model, along with training and testing.

Model architecture is as follows:
* Image descriptors of image keypoints are aggregated from image snippets created in Step 1 using OpenCV descriptor extractors (SIFT, SURF, ORB, etc).
* Set of image descriptors is clustered to form Bag of Visual Words (BOVW). Number of clusters can be specified, otherwise many different cluster sizes are tried and the optimal number of clusters (found via elbow method) is used.
* Histograms of BOVW are created from training image snippets. If desired, histograms of a reduced color space are also created and appended to the BOVW histograms.
* Histograms are used as input to model of choice (SVM, KNN, Logistic Regression). Before training/testing, histograms are transformed using the specified combination of data transformation (exponential scaling, normalization), feature selection (variance threshold, Chi^2), and approximation kernel mapping (RBF, Chi^2).


### Step 3: Entity Localization

(Implemented but not thoroughly tested.) Uses [Efficient Subwindow Search](https://ieeexplore.ieee.org/document/5166448) to locate entities within specified images.

`ess.py` implements the ESS algorithm.

`f_hat.py` implements the quality bounding function used by `ess.py`.

`localize.py` takes in a model from Step 2 and an image and tries to localize the various entities on which the model was trained.

