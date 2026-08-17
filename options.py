import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="AdaIR Training and Validation Options")

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=8, help="Batch size to use per GPU")
parser.add_argument('--val_batch_size', type=int, default=4, help="Batch size for validation")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--scale', type=int, default=2, help='Scale factor between input and target resolution (default: 2).')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataloader (default: 2).')

# Paths & Folder Mappings
parser.add_argument('--data_dir', type=str, default='data/train', help='root directory for training data')
parser.add_argument('--input_dir', type=str, default='NoisyLR', help='subfolder or directory name for degraded/input images (e.g. NoisyLR)')
parser.add_argument('--target_dir', type=str, default='GT', help='subfolder or directory name for ground truth/target clean images (e.g. GT)')

parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default=None, help='where clean images of denoising saves.')
parser.add_argument('--gopro_dir', type=str, default=None, help='where clean images of deblurring saves.')
parser.add_argument('--enhance_dir', type=str, default=None, help='where clean images of enhancement saves.')
parser.add_argument('--derain_dir', type=str, default=None, help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default=None, help='where training images of dehazing saves.')
parser.add_argument('--val_dir', type=str, default='data/train', help='directory containing validation image pairs.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument('--ckpt_dir', type=str, default="ckpt/", help='directory where checkpoints will be saved')
parser.add_argument('--metrics_file', type=str, default="training_metrics.csv", help='CSV file to save per-epoch validation metrics')
parser.add_argument("--wblogger", type=str, default=None, help="Determine to log to wandb or not and the project name")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")

options, _ = parser.parse_known_args()

# Dynamically resolve task subdirectories if data_dir is provided
if options.data_dir:
    data_path = Path(options.data_dir)
    def resolve_task_dir(task_name: str, fallback_default: str) -> str:
        for candidate in [task_name, task_name.lower(), task_name.capitalize()]:
            p = data_path / candidate
            if p.exists():
                return str(p) + '/'
        return str(data_path / task_name) + '/'

    if options.denoise_dir is None:
        options.denoise_dir = resolve_task_dir('Denoise', 'data/Train/Denoise/')
    if options.gopro_dir is None:
        options.gopro_dir = resolve_task_dir('Deblur', 'data/Train/Deblur/')
    if options.enhance_dir is None:
        options.enhance_dir = resolve_task_dir('Enhance', 'data/Train/Enhance/')
    if options.derain_dir is None:
        options.derain_dir = resolve_task_dir('Derain', 'data/Train/Derain/')
    if options.dehaze_dir is None:
        options.dehaze_dir = resolve_task_dir('Dehaze', 'data/Train/Dehaze/')
