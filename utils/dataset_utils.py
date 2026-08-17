import os
import random
import copy
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


def load_image_or_npy(path) -> np.ndarray:
    """
    Loads an image file (.png, .jpg, etc.) or a NumPy array (.npy) file
    and returns a 3-channel RGB numpy array (H, W, 3) in uint8 format.
    """
    path_str = str(path)
    if path_str.lower().endswith('.npy'):
        arr = np.load(path_str)
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
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    else:
        return np.array(Image.open(path_str).convert('RGB'))


class AdaIRTrainDataset(Dataset):
    def __init__(self, args):
        super(AdaIRTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

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
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        if os.path.exists(hazy):
            temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):
        temp_ids = []
        sub_input = getattr(self.args, 'input_dir', 'blur')
        target_dir = os.path.join(self.args.gopro_dir, sub_input)
        if not os.path.exists(target_dir):
            target_dir = os.path.join(self.args.gopro_dir, 'blur')
        if not os.path.exists(target_dir):
            target_dir = self.args.gopro_dir

        if os.path.exists(target_dir):
            image_list = filter_system_files(os.listdir(target_dir))
            temp_ids = image_list
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.deblur_ids = self.deblur_ids * 5
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        sub_input = getattr(self.args, 'input_dir', 'low')
        target_dir = os.path.join(self.args.enhance_dir, sub_input)
        if not os.path.exists(target_dir):
            target_dir = os.path.join(self.args.enhance_dir, 'low')
        if not os.path.exists(target_dir):
            target_dir = self.args.enhance_dir

        if os.path.exists(target_dir):
            image_list = filter_system_files(os.listdir(target_dir))
            temp_ids = image_list
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids * 20
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        if os.path.exists(rs):
            temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        scale = getattr(self.args, 'scale', 1)
        H = img_1.shape[0]
        W = img_1.shape[1]

        patch_size_in = self.args.patch_size
        patch_size_tg = self.args.patch_size * scale

        ind_H = random.randint(0, H - patch_size_in)
        ind_W = random.randint(0, W - patch_size_in)

        patch_1 = img_1[ind_H:ind_H + patch_size_in, ind_W:ind_W + patch_size_in]

        ind_H_tg = ind_H * scale
        ind_W_tg = ind_W * scale
        patch_2 = img_2[ind_H_tg:ind_H_tg + patch_size_tg, ind_W_tg:ind_W_tg + patch_size_tg]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(load_image_or_npy(clean_id), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(load_image_or_npy(sample["clean_id"]), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(load_image_or_npy(clean_name), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(load_image_or_npy(sample["clean_id"]), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(load_image_or_npy(clean_name), base=16)
            elif de_id == 5:
                # Deblur with Gopro set
                degrad_img = crop_img(load_image_or_npy(os.path.join(self.args.gopro_dir, 'blur/', sample["clean_id"])), base=16)
                clean_img = crop_img(load_image_or_npy(os.path.join(self.args.gopro_dir, 'sharp/', sample["clean_id"])), base=16)
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
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
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
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
            self.ids = []
            target_dir = os.path.join(self.args.derain_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.derain_path, 'input')
            name_list = filter_system_files(os.listdir(target_dir))
            self.ids += [os.path.join(target_dir, id_) for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            target_dir = os.path.join(self.args.dehaze_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.dehaze_path, 'input')
            name_list = filter_system_files(os.listdir(target_dir))
            self.ids += [os.path.join(target_dir, id_) for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            target_dir = os.path.join(self.args.gopro_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.gopro_path, 'input')
            name_list = filter_system_files(os.listdir(target_dir))
            self.ids += [os.path.join(target_dir, id_) for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            target_dir = os.path.join(self.args.enhance_path, sub_input)
            if not os.path.exists(target_dir): target_dir = os.path.join(self.args.enhance_path, 'input')
            name_list = filter_system_files(os.listdir(target_dir))
            self.ids += [os.path.join(target_dir, id_) for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        sub_gt = getattr(self.args, 'target_dir', 'target')
        sub_in = getattr(self.args, 'input_dir', 'input')
        if self.task_idx == 0:
            gt_name = degraded_name.replace(sub_in, sub_gt).replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split(sub_in)[0] + f'{sub_gt}/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace(sub_in, sub_gt).replace("input", "target")
        elif self.task_idx == 3:
            gt_name = degraded_name.replace(sub_in, sub_gt).replace("input", "target")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(load_image_or_npy(degraded_path), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
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

    
