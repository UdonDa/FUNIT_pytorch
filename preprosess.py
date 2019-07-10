from glob import glob
from sys import exit
import os
import argparse


def main(args):
    lists = []
    DIRs = glob(f"{args.image_dir}/*")
    for DIR in DIRs:
        filename = DIR.split("/")[-1]
        files = sorted(glob(f"{DIR}/*"))

        files = [f+'\n' for f in files]
        output_path = os.path.join(args.results_dir, filename)
        with open(f"{output_path}.txt", 'wt') as f:
            f.writelines(files)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/home/yanai-lab/horita-d/conf/cvpr2020/funit/data/food')
    parser.add_argument('--results_dir', type=str, default='./data/food_txt')

    args = parser.parse_args()
    main(args)