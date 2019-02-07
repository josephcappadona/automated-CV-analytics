import sys
import os

args = sys.argv

usage = "\nUSAGE:  python extract_frames.py VIDEO_FILEPATH [FFMPEG_ARGS]\n\nExample:  python extract_frames.py my_video.mp4 \"-vf fps=5\"\n"
if len(args) < 2:
    print(usage)
    exit()

video_filepath = args[1]
ffmpeg_args = ' '.join(args[2:])

# parse video filename
video_filename = video_filepath.split('/')[-1]
video_name = video_filename[:video_filename.rfind('.')]

# make output directory
output_dir = "output"
frames_dir = "%s/%s/frames" % (output_dir, video_name)
os.makedirs(frames_dir, exist_ok=True)

# build ffmpeg command
ffmpeg_command = "ffmpeg -i \"{}\" {} \"{}/image-%06d.png\" -hide_banner".format(video_filepath, ffmpeg_args, frames_dir)

# log ffmpeg command
command_filepath = frames_dir + '/command'
print('\nLogging ffmpeg command in \'%s\'...' % command_filepath)
with open(command_filepath, 'w+') as command_file:
    command_file.write(ffmpeg_command)

# run ffmpeg command
os.system("eval `cat \"%s\"`" % command_filepath)
print('\n\nFrames written to \'%s\'.\n' % frames_dir)
