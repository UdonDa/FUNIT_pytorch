from glob import glob
import argparse
import os
import random
import sys



def main(args):
    # Make dirs.
    os.makedirs(args.output_dir, exist_ok=True)
    style_dir = os.path.join(args.output_dir, "style")
    content_dir = os.path.join(args.output_dir, "content")
    os.makedirs(style_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)

    classes = [str(x) for x in range(256)]
    random.shuffle(classes)
    content_classes = classes[:224]
    style_classes = classes[224:]

    for filename in glob(f"{args.dataset_dir}/*"):
        
        if not os.path.isdir(filename):
            continue
        
        file_id = filename.split("/")[-1]
        if file_id in content_classes:
            os.symlink(filename, f"{content_dir}/{file_id}")

        elif file_id in style_classes:
            tmp_style_dir = os.path.join(style_dir, file_id)
            os.makedirs(tmp_style_dir, exist_ok=True)
            os.symlink(filename, f"{tmp_style_dir}/{file_id}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='UECFOOD256')
    parser.add_argument('--output_dir', type=str, default='./data/uecfood256')
    args = parser.parse_args()

    args.dataset_dir = os.path.join(os.getcwd(), args.dataset_dir)
    print(args.dataset_dir)

    main(args)