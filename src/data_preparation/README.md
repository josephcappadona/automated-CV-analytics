
### Usage

```
# extract frames from video
python src/data_preparation/extract_frames.py data/my_video.mp4 "-vf fps=1"

# apply static labels to frames
python src/data_preparation/apply_label_template.py output/my_video/entity_template.xml output/my_video/frames

# create subimage snippets
python src/data_preparation/create_snippets.py output/my_video/frames my_video
```

### Details

`extract_frames.py` uses `ffmpeg` to extract frames from a source video and writes them to a frames directory.

`apply_label_template.py` takes a static label template and applies it to all of the images in the specified folder. This is useful when an entity appears in the same location across all video frames.

`label_template.py` contains the library functions that `apply_label_template.py` uses to load and edit label templates.

`create_snippets.py` extracts the labeled subimages and writes them to a snippets directory.
