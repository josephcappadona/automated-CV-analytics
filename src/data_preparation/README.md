
### USAGE

```
# extract frames from video
python src/data_preparation/extract_frames.py data/my_video.mp4 "-vf fps=1"

# apply static labels to frames
python src/data_preparation/apply_label_template.py output/my_video/entity_template.xml output/my_video/frames

# create subimage snippets
python src/data_preparation/create_snippets.py output/my_video/frames my_video
```
