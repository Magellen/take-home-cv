from utils import *
import fire
import sys

if __name__ == "__main__":
    if not os.path.isfile("traffic.mp4"):
        print("missing video in current directory")
        sys.exit(0)
    if not os.path.exists('imgs'):
        v2img()
    if not os.path.isfile("heatmap.jpg"):
        createHeatMap()
    fire.Fire(run)
