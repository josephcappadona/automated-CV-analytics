import sys
import os
import utils


usage = \
'''USAGE:  python extract_frames.py VIDEO_FILEPATH [FFMPEG_ARGS]

Example:  python extract_frames.py my_video.mp4 "-vf fps=5"
'''

args = sys.argv
if len(args) < 2 or len(args) > 3:
    print(usage)
    exit()

video_filepath = args[1]
ffmpeg_args = args[2]

# parse video filename
video_filename = utils.get_filename(video_filepath)
video_name = utils.remove_extension(video_filename)

# make output directory
output_dir = "output"
frames_dir = os.path.join(output_dir, video_name, 'frames')
os.makedirs(frames_dir, exist_ok=True)

# build ffmpeg command
ffmpeg_command = "ffmpeg -i '%s' %s '%s/frame-%%06d.png' -hide_banner" \
                     % (video_filepath, ffmpeg_args, frames_dir)

# log ffmpeg command
command_filepath = os.path.join(frames_dir, 'command')
print('\nLogging ffmpeg command in \'%s\'...' % command_filepath)
with open(command_filepath, 'w+') as command_file:
    command_file.write(ffmpeg_command)

# run ffmpeg command
os.system(ffmpeg_command)
print('\n\nFrames written to \'%s\'.\n' % frames_dir)

