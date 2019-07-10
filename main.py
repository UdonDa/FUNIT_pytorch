import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from glob import glob

from sys import exit


def main(aegs):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    os.makedirs(args.results_dir    , exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Create Style Loaders and Iters.
    style_dirs = glob(f"{args.style_dir}/*")
    style_loaders = [get_loader(style_dir,
                        args.crop_size,
                        args.image_size,
                        args.batch_size,
                        args.mode,
                        args.num_workers) for style_dir in style_dirs]
    args.c_dim = len(style_loaders)
    print("num of classes: ", args.c_dim)

    # Create Content Loader.
    content_loader = get_loader(args.content_dir,
                        args.crop_size,
                        args.image_size,
                        args.batch_size,
                        args.mode,
                        args.num_workers)

    solver = Solver(content_loader, style_loaders, args)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    name = "debug"

    # Model argsuration.
    # parser.add_argument('--crop_size', type=int, default=286, help='crop size for the RaFD dataset')
    # parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=138, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training argsuration.
    parser.add_argument('--dataset', type=str, default='food')
    parser.add_argument('--c_dim', type=int, default=13, help='number of dataset classes.')
    parser.add_argument('--num_iters', type=int, default=200000, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='mini-batch size')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Test argsuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--content_dir', type=str, default='data/food_content')
    parser.add_argument('--style_dir', type=str, default='data/food_style')

    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--log_dir', type=str, default=f'results/{name}/logs')
    parser.add_argument('--model_save_dir', type=str, default=f'results/{name}/models')
    parser.add_argument('--sample_dir', type=str, default=f'results/{name}/samples')
    parser.add_argument('--result_dir', type=str, default=f'results/{name}/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)

    args = parser.parse_args()
    # print(args)
    main(args)