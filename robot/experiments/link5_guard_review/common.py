"""Hash, JSON and video inspection helpers for saved guard reviews."""
import json
import subprocess
from infrastructure.deformation_detect.coordinator import digest as sha, write

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,profile,pix_fmt:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], text=True))
