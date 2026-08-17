import os
import random
import copy
from pathlib import Path
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

IGNORED_SYSTEM_FILES = {'.ds_store', 'thumbs.db', 'desktop.ini', '.gitignore'}
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.npy')


def filter_system_files(file_list):
    """Filters out OS metadata files like .DS_Store, Thumbs.db, desktop.ini, and hidden files."""
    valid = []
    for f in file_list:
        basename = os.path.basename(f).lower()
        if basename.startswith('.') or basename.startswith('._'):
            continue
        if basename in IGNORED_SYSTEM_FILES:
            continue
        ext = os.path.splitext(basename)[1]
        if ext in VALID_EXTENSIONS:
            valid.append(f)
    return valid


def is_valid_image_file(path: Path) -> bool:
    """Returns True if path is a non-hidden, valid image/npy file."""
    name_lower = path.name.lower()
    if name_lower.startswith('.') or name_lower.startswith('._'):
        return False
    if name_lower in IGNORED_SYSTEM_FILES:
        return False
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def load_image_or_npy(path) -> np.ndarray:
    """
    Safely loads an image file or a NumPy array (.npy) file.
    - If path ends with .npy, loads using np.load(path).
    - Otherwise, loads using PIL.Image.open(path).convert('RGB').
    Returns a (H, W, 3) RGB numpy array in float32 [0.0, 1.0] range.
    """
    path_str = str(path)
    if path_str.lower().endswith('.npy'):
        arr = np.load(path_str).astype(np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[2] > 4:
                arr = arr.transpose(1, 2, 0)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
    else:
        arr = np.array(Image.open(path_str).convert('RGB'), dtype=np.float32)

    # Dynamic normalization: enforce strict float32 [0.0, 1.0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


class AdaIRTrainDataset(Dataset):
    def __init__(self, args):
        super(AdaIRTrainDataset, self).__init__()
        self.args = args
        self.scale = getattr(args, 'scale', 2)
        self.patch_size = getattr(args, 'patch_size', 128)
        self.sample_ids = []
        self.is_custom_paired = False

        data_dir = Path(getattr(args, 'data_dir', 'data/train'))
        input_sub = getattr(args, 'input_dir', 'NoisyLR')
        target_sub = getattr(args, 'target_dir', 'GT')

        input_path = self._find_subfolder(data_dir, [input_sub, 'NoisyLR', 'input', 'inputs'])
        target_path = self._find_subfolder(data_dir, [target_sub, 'GT', 'gt', 'target', 'targets'])

        if input_path and target_path and input_path.exists() and target_path.exists():
            self.is_custom_paired = True
            self._init_custom_paired_ids(input_path, target_path)
        else:
            # Fallback to benchmark datasets
            self.rs_ids = []
            self.hazy_ids = []
            self.D = Degradation(args)
            self.de_temp = 0
            self.de_type = self.args.de_type
            self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5, 'enhance': 6}
            self._init_ids()
            self._merge_ids()

        # Apply --max_samples subsampling if specified
        max_n = getattr(args, 'max_samples', None)
        if max_n is not None and max_n > 0 and len(self.sample_ids) > max_n:
            self.sample_ids = self.sample_ids[:max_n]
            print(f"[AdaIRTrainDataset] Subsampled to {max_n} pairs (--max_samples={max_n})")

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(self.patch_size),
        ])
        self.toTensor = ToTensor()

    def _find_subfolder(self, base_path: Path, candidates: list):
        for candidate in candidates:
            sub = base_path / candidate
            if sub.exists() and sub.is_dir():
                return sub
            if base_path.exists() and base_path.is_dir():
                for item in base_path.iterdir():
                    if item.is_dir() and item.name.lower() == candidate.lower():
                        return item
        return None

    def _init_custom_paired_ids(self, input_dir: Path, target_dir: Path):
        in_files = {f.name: f for f in input_dir.rglob('*') if is_valid_image_file(f)}
        tg_files = {f.name: f for f in target_dir.rglob('*') if is_valid_image_file(f)}

        matched = 0
        for name, in_p in in_files.items():
            tg_p = tg_files.get(name)
            if not tg_p:
                for tname, tp in tg_files.items():
                    if tp.stem == in_p.stem:
                        tg_p = tp
                        break
            if tg_p:
                self.sample_ids.append({
                    'degrad_path': str(in_p),
                    'clean_path': str(tg_p),
                    'name': in_p.stem,
                    'is_custom': True
                })
                matched += 1

        print(f"[AdaIRTrainDataset] Single-Task Paired Mode: Found {matched} matched pairs ({input_dir.name}/ <-> {target_dir.name}/)")

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()
        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        if os.path.exists(ref_file):
            temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        if os.path.exists(self.args.denoise_dir):
            name_list = filter_system_files(os.listdir(self.args.denoise_dir))
            clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if not temp_ids or id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids] * 3
            random.shuffle(self.s15_ids)
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids] * 3
            random.shuffle(self.s25_ids)
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids] * 3
            random.shuffle(self.s50_ids)

        self.num_clean = len(clean_ids)

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        if os.path.exists(hazy):
            temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.num_hazy = len(self.hazy_ids)

    def _init_deblur_ids(self):
        temp_ids = []
        sub_input = getattr(self.args, 'input_dir', 'blur')
        target_dir = os.path.join(self.args.gopro_dir, sub_input)
        if not os.path.exists(target_dir): target_dir = os.path.join(self.args.gopro_dir, 'blur')
        if not os.path.exists(target_dir): target_dir = self.args.gopro_dir

        if os.path.exists(target_dir):
            temp_ids = filter_system_files(os.listdir(target_dir))
        self.deblur_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids] * 5
        self.num_deblur = len(self.deblur_ids)

    def _init_enhance_ids(self):
        temp_ids = []
        sub_input = getattr(self.args, 'input_dir', 'low')
        target_dir = os.path.join(self.args.enhance_dir, sub_input)
        if not os.path.exists(target_dir): target_dir = os.path.join(self.args.enhance_dir, 'low')
        if not os.path.exists(target_dir): target_dir = self.args.enhance_dir

        if os.path.exists(target_dir):
            temp_ids = filter_system_files(os.listdir(target_dir))
        self.enhance_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids] * 20
        self.num_enhance = len(self.enhance_ids)

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        if os.path.exists(rs):
            temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids] * 120
        self.num_rl = len(self.rs_ids)

    def _crop_patch(self, img_1, img_2):
        """Crops input patch of size patch_size and target patch of size patch_size * scale.
        Default LR patch = 64x64, GT patch = 128x128 for 2x scale."""
        H, W = img_1.shape[0], img_1.shape[1]
        patch_size_in = self.patch_size
        patch_size_tg = self.patch_size * self.scale

        ind_H = random.randint(0, max(0, H - patch_size_in))
        ind_W = random.randint(0, max(0, W - patch_size_in))

        patch_1 = img_1[ind_H:ind_H + patch_size_in, ind_W:ind_W + patch_size_in]

        ind_H_tg = ind_H * self.scale
        ind_W_tg = ind_W * self.scale
        patch_2 = img_2[ind_H_tg:ind_H_tg + patch_size_tg, ind_W_tg:ind_W_tg + patch_size_tg]

        return patch_1, patch_2

    @staticmethod
    def _augment_pair(img_lr, img_hr):
        """8-fold paired augmentation: identical random hflip, vflip, and rotation
        applied simultaneously to both LR input and HR target arrays."""
        # Horizontal flip
        if random.random() > 0.5:
            img_lr = np.flip(img_lr, axis=1)
            img_hr = np.flip(img_hr, axis=1)
        # Vertical flip
        if random.random() > 0.5:
            img_lr = np.flip(img_lr, axis=0)
            img_hr = np.flip(img_hr, axis=0)
        # Random rotation from {0, 90, 180, 270}
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img_lr = np.rot90(img_lr, k=k)
            img_hr = np.rot90(img_hr, k=k)
        return np.ascontiguousarray(img_lr), np.ascontiguousarray(img_hr)

    def _get_gt_name(self, rainy_name):
        return rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]

    def _get_deblur_name(self, deblur_name):
        return deblur_name.replace("blur", "sharp")

    def _get_enhance_name(self, enhance_name):
        return enhance_name.replace("low", "gt")

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        return dir_name + name + suffix

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += getattr(self, 's15_ids', []) + getattr(self, 's25_ids', []) + getattr(self, 's50_ids', [])
        if "derain" in self.de_type:
            self.sample_ids += getattr(self, 'rs_ids', [])
        if "dehaze" in self.de_type:
            self.sample_ids += getattr(self, 'hazy_ids', [])
        if "deblur" in self.de_type:
            self.sample_ids += getattr(self, 'deblur_ids', [])
        if "enhance" in self.de_type:
            self.sample_ids += getattr(self, 'enhance_ids', [])

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        if sample.get('is_custom'):
            degrad_img = load_image_or_npy(sample['degrad_path'])  # float32 [0, 1]
            clean_img = load_image_or_npy(sample['clean_path'])    # float32 [0, 1]

            # Paired patch extraction (LR: patch_size, GT: patch_size * scale)
            degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

            # Low-variance filter: reject flat target patches (var < 1e-4), retry
            max_retries = 5
            retry = 0
            while clean_patch.var() < 1e-4 and retry < max_retries:
                degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)
                retry += 1

            # 8-fold paired augmentation (identical transforms to both)
            degrad_patch, clean_patch = self._augment_pair(degrad_patch, clean_patch)

            # Convert to tensors (already float32 [0, 1])
            clean_patch = torch.from_numpy(clean_patch.copy()).permute(2, 0, 1).float()
            degrad_patch = torch.from_numpy(degrad_patch.copy()).permute(2, 0, 1).float()

            return [sample['name'], 0], degrad_patch, clean_patch
        else:
            de_id = sample["de_type"]
            if de_id < 3:
                clean_id = sample["clean_id"]
                clean_img = crop_img(load_image_or_npy(clean_id), base=16)
                clean_patch = self.crop_transform(clean_img)
                clean_patch = np.array(clean_patch)
                clean_name = clean_id.split("/")[-1].split('.')[0]
                clean_patch = random_augmentation(clean_patch)[0]
                degrad_patch = self.D.single_degrade(clean_patch, de_id)
            else:
                if de_id == 3:
                    degrad_img = crop_img(load_image_or_npy(sample["clean_id"]), base=16)
                    clean_name = self._get_gt_name(sample["clean_id"])
                    clean_img = crop_img(load_image_or_npy(clean_name), base=16)
                elif de_id == 4:
                    degrad_img = crop_img(load_image_or_npy(sample["clean_id"]), base=16)
                    clean_name = self._get_nonhazy_name(sample["clean_id"])
                    clean_img = crop_img(load_image_or_npy(clean_name), base=16)
                elif de_id == 5:
                    degrad_img = crop_img(load_image_or_npy(os.path.join(self.args.gopro_dir, 'blur/', sample["clean_id"])), base=16)
                    clean_img = crop_img(load_image_or_npy(os.path.join(self.args.gopro_dir, 'sharp/', sample["clean_id"])), base=16)
                    clean_name = self._get_deblur_name(sample["clean_id"])
                elif de_id == 6:
                    degrad_img = crop_img(load_image_or_npy(os.path.join(self.args.enhance_dir, 'low/', sample["clean_id"])), base=16)
                    clean_img = crop_img(load_image_or_npy(os.path.join(self.args.enhance_dir, 'gt/', sample["clean_id"])), base=16)
                    clean_name = self._get_enhance_name(sample["clean_id"])

                degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

            clean_patch = self.toTensor(clean_patch)
            degrad_patch = self.toTensor(degrad_patch)

            return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15
        self._init_clean_ids()
        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = filter_system_files(os.listdir(self.args.denoise_path))
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(load_image_or_npy(self.clean_ids[clean_id]), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]
        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        sub_input = getattr(self.args, 'input_dir', 'input')
        if self.task_idx == 0:
            target_dir = os.path.join(self.args.derain_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.derain_path, 'input')
        elif self.task_idx == 1:
            target_dir = os.path.join(self.args.dehaze_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.dehaze_path, 'input')
        elif self.task_idx == 2:
            target_dir = os.path.join(self.args.gopro_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.gopro_path, 'input')
        elif self.task_idx == 3:
            target_dir = os.path.join(self.args.enhance_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.enhance_path, 'input')

        if os.path.exists(target_dir):
            name_list = filter_system_files(os.listdir(target_dir))
            self.ids = [os.path.join(target_dir, id_) for id_ in name_list]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        sub_gt = getattr(self.args, 'target_dir', 'target')
        sub_in = getattr(self.args, 'input_dir', 'input')
        if self.task_idx == 0:
            return degraded_name.replace(sub_in, sub_gt).replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split(sub_in)[0] + f'{sub_gt}/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            return dir_name + name
        else:
            return degraded_name.replace(sub_in, sub_gt).replace("input", "target")

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(load_image_or_npy(degraded_path), base=16)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(load_image_or_npy(clean_path), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = Path(degraded_path).stem

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)
        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'npy', 'NPY']
        if os.path.isdir(root):
            name_list = []
            for image_file in filter_system_files(os.listdir(root)):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image/npy files')
            self.degraded_ids += [os.path.join(root, id_) for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image/NPY file')
            self.degraded_ids = name_list
        print("Total Images/Arrays : {}".format(len(name_list)))
        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(load_image_or_npy(self.degraded_ids[idx]), base=16)
        name = Path(self.degraded_ids[idx]).stem
        degraded_img = self.toTensor(degraded_img)
        return [name], degraded_img

    def __len__(self):
        return self.num_img
