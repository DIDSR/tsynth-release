import torch
import pandas as pd
import os
import pickle
import h5py
import numpy as np
import warnings
import re
import random
import pydicom
import torchvision.transforms.functional as F
import config_global
from utils.duke_dbt_data import dcmread_image, read_labels
from PIL import Image, ImageOps, ImageDraw


def read_mhd(filename):
    if 'None' in filename:
        return None
    data = {}
    with open(filename, "r") as f:
        for line in f:
            s = re.search("([a-zA-Z]*) = (.*)", line)
            data[s[1]] = s[2]

            if " " in data[s[1]]:
                data[s[1]] = data[s[1]].split(" ")
                for i in range(len(data[s[1]])):
                    if data[s[1]][i].replace(".", "").replace("-", "").isnumeric():
                        if "." in data[s[1]][i]:
                            data[s[1]][i] = float(data[s[1]][i])
                        else:
                            data[s[1]][i] = int(data[s[1]][i])
            else:
                if data[s[1]].replace(".", "").replace("-", "").isnumeric():
                    if "." in data[s[1]]:
                        data[s[1]] = float(data[s[1]])
                    else:
                        data[s[1]] = int(data[s[1]])
    return data


def read_loc(filename):
    """returns coords as [y position, x position, slice number]"""
    with open(filename, "r") as file:
        lines = file.readlines()
    coords = [int(v) for v in lines[0].strip().split(" ")[0:3]]
    return coords


def get_lesion_box(x, y, box_side=150, image_width=1324, image_height=1624, SL=10, invert_y=True):
    y = image_height - y if invert_y else y
    
    x1 = max(0, x - box_side)
    y1 = max(0, y - box_side)
    x2 = min(image_width, x + box_side + SL)
    y2 = min(image_height, y + box_side + SL)

    box = [x1, y1, x2, y2]

    return box


def window_image(image, window_min, window_max):
    image = image.astype(np.float32) 
    image[image > window_max] = window_max
    image[image < window_min] = window_min
    image = (image - window_min) / (window_max - window_min + 1e-5)
    return image


def window_image_2(img, low=None, high=None):
    if low is None or high is None:
        low = np.percentile(img, 15)
        high = np.percentile(img, 99.5)

    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-5)
    img = (img * 255).astype(np.uint8)
    return img



def get_box_from_mask(lesion_mask, x_is_W=True):
    """
    Computes the bounding box for a binary mask.

    Args:
        lesion_mask (np.ndarray): A binary mask, either in the form (H, W) or (H, W, D).
        x_is_W (bool): If true, x and y correspond to W and H respectively (torchvision convention).
                       If false, x and y correspond to H and W respectively (monai convention).
        
    Returns:
        list: Bounding box coordinates.
              - For 2D: [x1, y1, x2, y2]
              - For 3D: [x1, y1, z1, x2, y2, z2]
    """
    # Find the indices where the mask is non-zero (1s)
    coords = np.argwhere(lesion_mask)
    
    # Get the minimum and maximum indices for each dimension
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    # Switch x and y if x_is_W
    if x_is_W:
        min_coords[0], min_coords[1] = min_coords[1], min_coords[0]
        max_coords[0], max_coords[1] = max_coords[1], max_coords[0]
    
    # Bounding box: [min, max] for each dimension
    bounding_box = np.concatenate([min_coords, max_coords], axis=0)
    
    return bounding_box


class BcsDbtDataset(torch.utils.data.Dataset):
    """Detection Dataset for BCS-DBT Data"""
    def __init__(self,
                 data_dir='/projects01/didsr-aiml/common_data/BCS-DBT/data/raw/',
                 image_dir='training_set_images/',
                 path_file='BCS-DBT file-paths-train-v2.csv',
                 boxes_file='BCS-DBT boxes-train-v2.csv',
                 label_file='BCS-DBT labels-train-v2.csv',
                 lesion_only=True):
        self.label_df = read_labels(*[os.path.join(data_dir, fp) for fp in [path_file, boxes_file, label_file]])
        if lesion_only:
            self.label_df = self.label_df[(self.label_df['Benign'] == 1) | (self.label_df['Cancer'] == 1)]
        self.image_dir = os.path.join(data_dir, image_dir)
        warnings.filterwarnings("ignore", message="The .* Bits Stored value '10' .*")
        
    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        box_series = self.label_df.iloc[idx]
        view = box_series["View"]
        slice_index = box_series["Slice"]
        image_path = os.path.join(self.image_dir, box_series["descriptive_path"])
        image = dcmread_image(fp=image_path, view=view, index=slice_index)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = image / image.max()
        
        x, y, width, height = box_series[["X", "Y", "Width", "Height"]]
        if np.isnan(x):
            boxes, labels = np.zeros((0, 4)), []
        else:
            boxes, labels = [[x, y, x+width, y+height]], [1]
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return image, target

def filter_one_image_per_patient(df, prefer_positive=True, random_seed=42):
    """
    From each patient (empi_anon), select one image.
    If prefer_positive=True, prioritize selecting an image with ROI if available.
    """
    np.random.seed(random_seed)
    selected_entries = []

    for empi, group in df.groupby('empi_anon'):
        if prefer_positive and (group['num_roi'] > 0).any():
            group_with_finding = group[group['num_roi'] > 0]
            selected = group_with_finding.sample(n=1)
        else:
            selected = group.sample(n=1)
        selected_entries.append(selected)

    return pd.concat(selected_entries).reset_index(drop=True)

def resize_and_pad_pair(image, mask, target_size, fill_image=(0, 0, 0), fill_mask=0):
    """
    Pad to square (right/bottom only), then resize to target.
    """
    orig_w, orig_h = image.size
    original_size = (orig_w, orig_h)
    max_dim = max(orig_w, orig_h)

    pad_right = max_dim - orig_w
    pad_bottom = max_dim - orig_h

    image_square = ImageOps.expand(image, (0, 0, pad_right, pad_bottom), fill=fill_image)
    mask_square = ImageOps.expand(mask, (0, 0, pad_right, pad_bottom), fill=fill_mask)

    scale = target_size[0] / max_dim  # assume square target
    resized_image = image_square.resize(target_size, Image.Resampling.LANCZOS)
    resized_mask = mask_square.resize(target_size, Image.Resampling.NEAREST)

    transform_info = {
        'original_size': original_size,
        'square_size': max_dim,
        'pad': (0, 0, pad_right, pad_bottom),
        'scale': scale
    }

    return resized_image, resized_mask, transform_info

class EmbedDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing the EMBED dataset.

    This class reads image metadata and clinical data from CSV files, filters the data based on specified criteria,
    and prepares the dataset for use in training or evaluation of machine learning models.

    Parameters:
    -----------
    root_dir : str
        The root directory where the dataset is located. Default is '/projects01/didsr-aiml/common_data/EMBED'.
    
    FinalImageType : list of str
        A list specifying the types of final images to include in the dataset.
        To use both mammogramns and cview, set to ['2D', 'cview']. Default is ['cview'].
    
    existing_paths : str or None
        A path to a python set (pickle file) that contains all available image paths. If not None, then the dataset will
        be filtered to contain only file paths in this set. This is needed if not all the files listed in
        EMBED_OpenData_metadata.csv are downloaded on the machine.
    
    data_split_dict : dict or None
        A dictionary containing parameters for splitting the dataset. This is useful for splitting the data into 
         train, validate, and/or test sets based on this csv. If provided, must include keys:
        - 'csv_file': Path to a CSV file containing the split information. This csv should have all anon_dicom_paths and the
        keyword of the set assigned to them.
        - 'keyword': A keyword (e.g. 'train', 'val', 'test') used to filter the dataset based on a specific group in the split CSV.

    sample_balance : str or None
        Used to specify balance between positive (samples with at least one ROI) and negative samples.
        - If None, then available data are used.
        - If 'balanced', then negative sample count is reduced to equal positive sample count.
        - If 'positive', then only positive samples are used.
        - If 'negative', then only negative samples are used.

    percent_samples_included: float32 or None
        Specifies the percentage of the total available data to use. Example: if set to 0.8, then a random 20% of the samples will be
        dropped from the dataset. Must be between 0-1. If None, then no samples will be excluded (equivalent as setting to 1.0).

    one_image_per_patient:
        Choose one image per patient.

    x_is_W : bool 
            Dictates the dimension assignment convention for bbox elements
            If True, x and y correspond to W and H respectively (torchvision convention).
            If False, x and y correspond to H and W respectively (monai convention).

        
    Returns:
    --------
    image: torch.tensor float32 of shape [C, H, W].
    target: dict with the following keys:
        labels: torch.tensor float64 1D array of 1s with length equal to the number of findings. 
        boxes: torch.tensor float64 array of finding bounding boxes, each in the form [x1, y1, x2, y2].
    """
    def __init__(self,
                 root_dir='/projects01/didsr-aiml/common_data/EMBED',
                 FinalImageType=['cview'],
                 existing_paths=None,
                 data_split_dict=None,
                 sample_balance=None,
                 percent_samples_included=None,
                 one_image_per_patient = False,
                 num_positive=None,
                 num_negative=None,  
                 x_is_W = True,
                 image_resize = None):
        
        self.root_dir = root_dir
        self.sample_balance = sample_balance
        self.FinalImageType = FinalImageType
        self.percent_samples_included = percent_samples_included
        self.x_is_W = x_is_W
        self.image_resize = image_resize
        self.num_positive = num_positive
        self.num_negative = num_negative

        csv_path = f'{self.root_dir}/tables/EMBED_OpenData_metadata_reduced.csv'
        self.label_df = pd.read_csv(csv_path, dtype={27: str, 28: str})
        self.label_df = self.label_df.drop(columns=['study_date_anon', 'StudyDescription', 'SeriesDescription',
                                                      'SRC_DST', 'match_level', 'SeriesNumber', 'SeriesTime',
                                                      'has_pix_array', 'category', 'ProtocolName'])
        # Add BD information from clinical data
        clinical_df = pd.read_csv(f'{self.root_dir}/tables/EMBED_OpenData_clinical.csv',
                                  dtype={key: str for key in [28,31,33,48,49,53,54,55,56,57,59,81,84,85,93,111]})
        density_df =clinical_df[['acc_anon', 'tissueden']]
        density_df = density_df.rename(columns={'tissueden': 'breast_density'})
        density_df = density_df.drop_duplicates()
        self.label_df = self.label_df.merge(density_df, how='inner', on='acc_anon')

        # Filter to files that exist
        self.label_df['anon_dicom_path'] = self.label_df['anon_dicom_path'].str.replace(
            '/mnt/NAS2/mammo/anon_dicom/', f'{self.root_dir}/images/', regex=False)
        if existing_paths is None:
            # existing_paths = set()
            # for root, _, files in os.walk(f'{self.root_dir}/images/'):
            #     for file in files:
            #         path = os.path.join(root, file)
            #         existing_paths.add(path)
            # with open('existing_Embed_fileset.pkl', 'wb') as f:
            #     pickle.dump(existing_paths, f)
            existing_paths = set(self.label_df['anon_dicom_path'].tolist())
        else:
            with open(existing_paths, 'rb') as f:
                existing_paths = pickle.load(f)
            self.label_df = self.label_df[self.label_df['anon_dicom_path'].isin(existing_paths)]

        # Filter the DataFrame based on FinalImageType
        self.label_df = self.label_df[self.label_df['FinalImageType'].isin(self.FinalImageType)]

        # Add clinical details for each ROI; modified from: https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/Sample_Notebook.ipynb
        positive_df = self.label_df[self.label_df['num_roi'] > 0]
        for idx, row in positive_df.iterrows():
            clinical_row = clinical_df[clinical_df['acc_anon'] == row['acc_anon']]

            clinical_row = clinical_row[
                (clinical_row['side'] == row['ImageLateralityFinal']) |
                (clinical_row['side'] == 'B') |
                (pd.isna(clinical_row['side']))]

            for _, roi_data in clinical_row.iterrows():
                if (roi_data['massshape'] in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']) or \
                   (roi_data['massmargin'] in ['D', 'U', 'M', 'I', 'S']) or \
                   (roi_data['massdens'] in ['+', '-', '=']):
                    self.label_df.at[idx, 'mass'] = 1        
                if roi_data['massshape'] in ['T', 'B', 'S', 'F', 'V']:
                    self.label_df.at[idx, 'asymmetry'] = 1
                if roi_data['massshape'] in ['Q', 'A']:
                    self.label_df.at[idx, 'arch_distortion'] = 1                  
                if not pd.isna(roi_data['calcdistri']) or \
                   not pd.isna(roi_data['calcfind']) or \
                   roi_data['calcnumber'] != 0:
                    self.label_df.at[idx, 'calc'] = 1

        # Split dataset by keyword if necessary
        if data_split_dict is not None:
            split_df = pd.read_csv(data_split_dict['csv_file'])
            split_empi_anon = split_df[split_df['group'] == data_split_dict['keyword']]['empi_anon'].tolist()
            self.label_df = self.label_df[self.label_df['empi_anon'].isin(split_empi_anon)]
            self.label_df.reset_index(drop=True, inplace=True)
        
        # Reduce to one image per patient
        if one_image_per_patient:
            self.label_df = filter_one_image_per_patient(self.label_df, prefer_positive=True)
            
            if self.num_positive is not None or self.num_negative is not None:
                pos_df = self.label_df[self.label_df['num_roi'] > 0]
                neg_df = self.label_df[self.label_df['num_roi'] == 0]

                if self.num_positive is not None:
                    pos_df = pos_df.sample(n=min(len(pos_df), self.num_positive), random_state=42)
                if self.num_negative is not None:
                    neg_df = neg_df.sample(n=min(len(neg_df), self.num_negative), random_state=42)

                self.label_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        # Remove entries window center isn't working
        # for idxx, entry in self.label_df.iterrows():
        #     try:
        #         win_center, win_half_width = eval(entry['WindowCenter']), eval(entry['WindowWidth'])/2.
        #     except:
        #         print(entry['WindowCenter'])
        #         print(eval(entry['WindowWidth']))

        # Filter based on finding_only
        if self.sample_balance is not None:
            if self.sample_balance == 'positive':
                self.label_df = self.label_df[self.label_df['num_roi'] > 0]
            elif self.sample_balance == 'negative':
                self.label_df = self.label_df[self.label_df['num_roi'] == 0]

            elif self.sample_balance == 'balanced':
                positive_samples = self.label_df[self.label_df['num_roi'] > 0]
                negative_samples = self.label_df[self.label_df['num_roi'] == 0]
                negative_samples_keep = []
                for bd in [1.0, 2.0, 3.0, 4.0]:
                    negative_samples_bd = negative_samples[negative_samples['breast_density'] == bd]
                    positive_samples_bd = positive_samples[positive_samples['breast_density'] == bd]
                    if len(negative_samples_bd) > len(positive_samples_bd):
                        negative_samples_bd = negative_samples_bd.sample(n=len(positive_samples_bd), random_state=42)
                    negative_samples_keep.append(negative_samples_bd)
                negative_samples_keep = pd.concat(negative_samples_keep, ignore_index=True)
                self.label_df = pd.concat([positive_samples, negative_samples_keep], ignore_index=True)
            else:
                assert False, 'Error: self.sample_balance must be one of the following: positive, negative, balanced'
            self.label_df.reset_index(drop=True, inplace=True)

        # Reduce total number of samples if specified:
        if self.percent_samples_included is not None:
            self.label_df = self.label_df.sample(frac=self.percent_samples_included)
            self.label_df.reset_index(drop=True, inplace=True)


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.label_df)

    # Get DICOM image metadata. From https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/DCM_to_PNG.ipynb for EMBED image processing.
    class DCM_Tags():
        def __init__(self, img_dcm):
            try:
                self.laterality = img_dcm.ImageLaterality
            except AttributeError:
                self.laterality = np.nan
                
            try:
                self.view = img_dcm.ViewPosition
            except AttributeError:
                self.view = np.nan
                
            try:
                self.orientation = img_dcm.PatientOrientation
            except AttributeError:
                self.orientation = np.nan

    # Check whether DICOM should be flipped. From https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/DCM_to_PNG.ipynb for EMBED image processing.
    def check_dcm(self, imgdcm):
        # Get DICOM metadata
        tags = self.DCM_Tags(imgdcm)
        
        # If image orientation tag is defined
        if not pd.isnull(tags.orientation):
            # CC view
            if tags.view == 'CC':
                if tags.orientation[0] == 'P':
                    flipHorz = True
                else:
                    flipHorz = False
                
                if (tags.laterality == 'L') & (tags.orientation[1] == 'L'):
                    flipVert = True
                elif (tags.laterality == 'R') & (tags.orientation[1] == 'R'):
                    flipVert = True
                else:
                    flipVert = False
            
            # MLO or ML views
            elif (tags.view == 'MLO') | (tags.view == 'ML'):
                if tags.orientation[0] == 'P':
                    flipHorz = True
                else:
                    flipHorz = False
                
                if (tags.laterality == 'L') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HL')):
                    flipVert = True
                elif (tags.laterality == 'R') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HR')):
                    flipVert = True
                else:
                    flipVert = False
            
            # Unrecognized view
            else:
                flipHorz = False
                flipVert = False
                
        # If image orientation tag is undefined
        else:
            # Flip RCC, RML, and RMLO images
            if (tags.laterality == 'R') & ((tags.view == 'CC') | (tags.view == 'ML') | (tags.view == 'MLO')):
                flipHorz = True
                flipVert = False
            else:
                flipHorz = False
                flipVert = False
                
        return flipHorz, flipVert

    def __getitem__(self, idx):
        entry = self.label_df.iloc[idx]

        # Load the DICOM image
        ds = pydicom.dcmread(entry['anon_dicom_path'])
        flipHorz, flipVert = self.check_dcm(ds)
        image = ds.pixel_array  # Extract pixel data
        image = np.fliplr(image) if flipHorz else image
        image = np.flipud(image) if flipVert else image

        try:
            win_center, win_half_width = eval(entry['WindowCenter']), eval(entry['WindowWidth']) / 2.
        except:
            win_center, win_half_width = 540.0, 290.0  # fallback    
        
        roi_coords_list = [i for i in eval(entry['ROI_coords']) if len(i) == 4]
        boxes = np.array(roi_coords_list).reshape(-1, 4)
        if self.x_is_W:
            boxes = boxes[:, [1, 0, 3, 2]] 
        
        if self.image_resize == True:

            image = window_image_2(image)
            image_pil = Image.fromarray(image).convert("L")

            mask = Image.fromarray(np.zeros_like(image, dtype=np.uint8))
            for box in boxes:
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(list(box), fill=1)

            # Resize image and mask
            resized_image_pil, resized_mask_pil, transform_info = resize_and_pad_pair(image_pil, mask, (512, 512),0)
            image = F.to_tensor(resized_image_pil)
            scale = transform_info['scale']
            pad = transform_info['pad']
            boxes_rescaled = boxes + np.array([pad[0], pad[1], pad[0], pad[1]])  # no left/top pad here, so zero
            boxes_rescaled = boxes_rescaled * scale

        else:
            image = window_image(image, win_center-win_half_width, win_center+win_half_width)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            boxes_rescaled = boxes  

        labels = np.ones(len(boxes_rescaled))
        target = {
            'boxes': torch.tensor(boxes_rescaled, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

class ZeroPadTransform():
    """Pads input image on top, bottom, and right such that the output image [height, width] is target_size, regardless of input size.
    Note: This has only been designed for 2D images so far."""
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        self.target_height, self.target_width = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, target):
        # Get original size of the image
        orig_height, orig_width = image.shape[-2:]

        # Calculate padding
        pad_top = max(0, (self.target_height - orig_height) // 2)
        pad_bottom = max(0, self.target_height - orig_height - pad_top)
        pad_right = max(0, self.target_width - orig_width)

        # Apply padding to the image
        padded_image = F.pad(image, (0, pad_top, pad_right, pad_bottom),
                             fill=self.fill, padding_mode=self.padding_mode)

        # Adjust bounding box
        # bbox format: [x_min, y_min, x_max, y_max]
        for i in range(len(target['boxes'])):
            target['boxes'][i][1] = target['boxes'][i][1] + pad_top
            target['boxes'][i][3] = target['boxes'][i][3] + pad_top

        return padded_image, target


class ChangePixelSizeTransform():
    """Changes pixel spacing of 2D image from old_pixel_size to new_pixel_size via image resizing."""
    def __init__(self, old_pixel_size, new_pixel_size):
        self.scale = old_pixel_size / new_pixel_size

    def __call__(self, image, target):
        # Get original size of the image
        orig_height, orig_width = image.shape[-2:]
        new_height, new_width = int(orig_height * self.scale), int(orig_width * self.scale)

        # Resize Image
        resized_image = F.resize(image, (new_height, new_width))

        # Adjust bounding box
        # bbox format: [x_min, y_min, x_max, y_max]
        if 'boxes' in target:
            target['boxes'] = target['boxes'] * self.scale

        return resized_image, target


class DbtSynthDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading and processing synthetic DBT (Digital Breast Tomosynthesis) images.

    This class is designed to work with synthetic DBT datasets, allowing for flexible configurations
    regarding lesion densities, breast densities, lesion sizes, and more. It extends the PyTorch Dataset
    class for easy integration into data loading pipelines.

    Parameters:
        root_dir (str): The root directory where the synthetic data are stored.
        lesion_densities (list): A list of lesion densities (str) to include.
        breast_densities (list): A list of breast densities (e.g. 'fatty') to include.
        lesion_sizes (list): A list of lesion sizes (str) in mm to include.
        patient_range (list): Two element list for of the range of patient identifiers to use.
        lesion_only (bool): If True, only samples with a lesion will be included.
        image_mode (str): The mode for image selection ('cview', 'random_slice', 'middle_slice', 'volume').
            cview: loads the 2D cview version of the image, which are located in cview_dir
            random_slice: For negative sample, returns random slice in the volume. For positive, returns random slice with lesion
            middle_slice: Returns the middle slice of the volume
        x_is_W (bool): If True, x and y correspond to W and H respectively (torchvision convention).
            If False, x and y correspond to H and W respectively (monai convention).
        window (optional): Any additional parameter for windowing (default is None).

    Returns:
        image: torch.tensor float32 of shape [C, H, W] or [C, H, W, D] if image_mode=volume
        target: dict with the following keys:
            labels: torch.tensor float64 [1] if image has lesion, [] if no lesion
            boxes: torch.tensor float64 array of lesion bounding boxes, in form [x1, y1, x2, y2] or [x1, y1, z1, x2, y2, z2] if image_mode=volume
            mask (if return_mask=True): torch.tensor bool lesion mask, with the same shape as image
    """
    def __init__(self,
                 root_dir=config_global.dir_global + 'data/cview/output_cview_det_Victre/',
                 lesion_densities=['1.06'],
                 breast_densities=['fatty'],
                 lesion_sizes=['5.0', '7.0', '9.0'],
                 patient_range = [1, 301],
                 lesion_only=True,
                 image_mode='volume',  # or random_slice or middle_slice or volume
                 return_mask=False,
                 x_is_W = True,
                 window=None,
                 padding=None,
                 num_positive=None,
                 num_negative=None,
                 transforms=None):

        # Define working values
        self.root_dir = root_dir
        self.lesion_densities = lesion_densities
        self.breast_densities = breast_densities
        self.lesion_sizes = lesion_sizes
        self.lesion_only = lesion_only
        self.return_mask = return_mask
        self.image_mode = image_mode
        self.x_is_W = x_is_W
        self.window = window
        self.padding = ZeroPadTransform(**padding) if padding is not None else None
        self.transforms = transforms
        self.num_positive = num_positive
        self.num_negative = num_negative
        
        # # Map lesion sizes to bbox sizes with dictionary
        # self.bbox_size_dict = {lesion_sizes[i]: bbox_sizes[i] for i in range(len(lesion_sizes))}
        # # Map lesion sizes to lesion radius in terms of slice count
        # self.lesion_width_dict = {lesion_sizes[i]: lesion_widths[i] for i in range(len(lesion_sizes))}
        # Map strings to ACR Breast Density scores
        acrs = {
            "fatty": 1,
            "scattered": 2,
            "hetero": 3,
            "dense": 4
        }

        self.tissue_lookup = {
            0: 'air',
            1: 'fat',
            2: 'skin',
            29: 'glandular',
            33: 'nipple',
            40: 'muscle',
            88: 'ligament',
            95: 'TDLU',
            125: 'duct',
            150: 'artery',
            225: 'vein',
            200: 'lesion',  # 'cancerous mass',
            250: 'calcification',
            50: 'compression paddle'
        }
        self.tissue_lookup = {v: k for k, v in self.tissue_lookup.items()}

        # Define the folder containing the images
        files = []
        for ld in lesion_densities:
            for bd in breast_densities:
                for ls in lesion_sizes:
                    file_path = os.path.join(self.root_dir, f'device_data_VICTREPhantoms_spic_{ld}/{bd}/2/{ls}/SIM')
                    if os.path.exists(file_path) and os.path.isdir(file_path):
                        for patient_idx in range(patient_range[0], patient_range[1]):
                            raw_path = os.path.join(file_path,
                                                    f'D2_{ls}_{bd}.{patient_idx}/{patient_idx}/reconstruction{patient_idx}.raw')
                            if os.path.exists(raw_path):
                                # check is loc file exists (has lesion)
                                has_lesion = os.path.exists(raw_path.replace('.raw', '.loc'))
                                if has_lesion or not lesion_only:
                                    files.append({
                                        'raw_path': raw_path,
                                        'breast_density': acrs[bd],
                                        'lesion_density': ld,
                                        'lesion_size': ls,
                                        'has_lesion': has_lesion
                                    })
                            else:
                                with open('missing_dbt.txt', 'a') as f:
                                    f.write(raw_path + '\n')

        self.label_df = pd.DataFrame(files)
        if self.num_positive is not None or self.num_negative is not None:
            pos_df = self.label_df[self.label_df['has_lesion']]
            neg_df = self.label_df[~self.label_df['has_lesion']]
            
            if self.num_positive is not None:
                pos_df = pos_df.sample(n=min(self.num_positive, len(pos_df)), random_state=42)
            if self.num_negative is not None:
                neg_df = neg_df.sample(n=min(self.num_negative, len(neg_df)), random_state=42)

            self.label_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.label_df)

    def get_image(self, raw_path):
        mhd_path = raw_path.replace('.raw', '.mhd')
        im_dims = read_mhd(mhd_path)['DimSize']
        im_height, im_width, total_slices = int(im_dims[0]), int(im_dims[1]), int(im_dims[2])

        # Retrieve Image
        slice_number = None
        if self.image_mode == 'cview':
            image = np.fromfile(raw_path, dtype="float32").reshape(im_height, im_width).astype(np.float32)
            image = window_image(image, image.min(), image.max())
        else:
            image = np.fromfile(raw_path, dtype="float64").reshape(total_slices, im_width, im_height).astype(np.float32)
            image = image.transpose(2, 1, 0)
            image = image[::-1]
            if self.image_mode == 'random_slice':
                slice_number = random.randint(0, total_slices)
                image = image[:, :, slice_number]
            elif self.image_mode == 'middle_slice':
                slice_number = int(total_slices / 2)
                image = image[:, :, slice_number]
            elif self.image_mode != 'volume':
                raise ValueError("Invalid image_model. Must be 'random_slice', 'middle_slice', 'cview', or 'volume'")
            X = np.std(image) * 2
            TH = np.mean(image)
            image[image < TH - X] = 0
            image[image > TH + X] = X + TH
            image = (image - image.min()) / (image.max() - image.min())

        if self.window is not None:
            image = window_image(image, *self.window)
        return image, slice_number

    def get_mask(self, mask_path, slice_number=None):
        labels = []
        boxes = np.zeros((0, 6)) if self.image_mode == 'volume' else np.zeros((0, 4))

        with h5py.File(mask_path, 'r') as h5_file:
            voxel_data = h5_file['dbt'][...]
        lesion_mask = voxel_data == self.tissue_lookup['lesion']

        if self.image_mode != 'cview':
            lesion_mask = lesion_mask.transpose(2, 1, 0)
            lesion_mask = lesion_mask[::-1]

        bbox = get_box_from_mask(lesion_mask, x_is_W=self.x_is_W)

        if slice_number is not None:
            lesion_mask = lesion_mask[:, :, slice_number]
            x1, y1, z1, x2, y2, z2 = bbox
            if z1 < slice_number < z2:
                bbox = np.array([x1, y1, x2, y2])
            else:
                bbox = None

        if bbox is not None:
            boxes = np.concatenate([boxes, np.expand_dims(bbox, axis=0)], axis=0)
            labels.append(1)
        return lesion_mask, boxes, labels

    def __getitem__(self, idx):
        box_series = self.label_df.iloc[idx]
        raw_path = box_series["raw_path"]
        image, slice_number = self.get_image(raw_path)

        # Get mask and bbox information
        labels = []
        boxes = np.zeros((0, 6)) if self.image_mode == 'volume' else np.zeros((0, 4))
        lesion_mask = np.zeros(image.shape, dtype=np.float32)
        mask_path = raw_path.replace('.raw', '.h5').replace('reconstruction', 'reconstruction_mask')
 
        if os.path.exists(mask_path) and box_series['has_lesion']:
            lesion_mask, boxes, labels = self.get_mask(mask_path, slice_number)

        # Create target dict and prepare image
        image = torch.tensor(image.copy(), dtype=torch.float32).unsqueeze(0)
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        if self.return_mask:
            lesion_mask = torch.tensor(lesion_mask.copy(), dtype=torch.float32).unsqueeze(0)
            target['mask'] = lesion_mask

        if self.padding is not None:
            image, target = self.padding(image, target)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class MSynthDetectionDataset(torch.utils.data.Dataset):
    """
    Similar to DBTSynthDataset, and modified from msnth-scracth/analysis/dataset_loader.datasets.MsynthDataset

    Parameters:
        root_dir (str): The root directory where the synthetic data are stored.
        lesion_densities (list): A list of lesion densities (str) to include.
        breast_densities (list): A list of breast densities (e.g. 'fatty') to include.
        lesion_sizes (list): A list of lesion sizes (str) in mm to include.
        patients (list): A list of patient identifiers to include.
        bbox_sizes (list): A list of bounding box sizes (side length in pixels) corresponding to each size in lesion_sizes.
        lesion_only (bool): If True, only samples with a lesion will be included.
        window (optional): Any additional parameter for windowing (default is None).
    """
    def __init__(self,
                 root_dir="/projects01/didsr-aiml/common_data/MSYNTH_data/",
                 bounds_dir = "/projects01/VICTRE/elena.sizikova/code/breast/realysm/example_generation/data/bounds/",
                 lesion_densities=['1.06'],
                 breast_densities=['fatty'],
                 lesion_sizes=['5.0', '7.0', '9.0'],
                 patient_range = [1, 301],
                 #patients=[str(patient) for patient in range(1, 301)],
                 bbox_sizes=[110, 150, 195],
                 proj_mode = True,
                 lesion_only=True,
                 window=None,
                 padding=None,
                 transforms=None):

        # Define working values
        self.root_dir = root_dir
        self.lesion_densities = lesion_densities
        self.breast_densities = breast_densities
        self.lesion_sizes = lesion_sizes
        self.lesion_only = lesion_only
        self.proj_mode = proj_mode
        self.window = window
        self.padding = ZeroPadTransform(**padding) if padding is not None else None
        self.transforms = transforms
        # Map lesion sizes to bbox sizes with dictionary
        self.bbox_size_dict = {lesion_sizes[i]: bbox_sizes[i] for i in range(len(lesion_sizes))}
        # Map strings to ACR Breast Density scores
        self.acrs = {
            "fatty": 1,
            "scattered": 2,
            "hetero": 3,
            "dense": 4
        }
         # Proper full doses for lesion
        doses_dct = {
            "fatty": "2.22e10",
            "scattered": "2.04e10",
            "hetero": "1.02e10",
            "dense": "8.67e09"
        }
        # Create dict for image bounds
        self.bounds_dict = {self.acrs[bd]: np.load(f'{bounds_dir}/bounds_{bd}.npy', allow_pickle=True)
               for bd in breast_densities}
        # Define the folder containing the images
        files = []
        for ld in lesion_densities:
            for bd in breast_densities:
                dose = doses_dct[bd]
                for ls in lesion_sizes:
                    folder_path = os.path.join(self.root_dir, f'data/device_data_VICTREPhantoms_spic_{ld}/{dose}/{bd}/2/{ls}/SIM')
                    mask_folder_path = os.path.join(self.root_dir, f'segmentation_masks/{bd}/2/{ls}/SIM')
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        for patient_idx in  range(patient_range[0], patient_range[1]):
                        #patients:

                            #folder = [filename for filename in os.listdir(folder_path) if filename.endswith(patient_idx)][0]
                            folder = [filename for filename in os.listdir(folder_path) if filename.split('.')[-1]==str(patient_idx)][0]
                            raw_path = os.path.join(folder_path, folder, str(patient_idx), f'projection_DM{patient_idx}.raw')
                            mask_path = os.path.join(mask_folder_path, f'P2_{ls}_{bd}.{patient_idx}/seg/{patient_idx}/projection_DM{patient_idx}.raw')
                            if os.path.exists(raw_path):
                                # check is loc file exists (has lesion)
                                has_lesion = os.path.exists(raw_path.replace('.raw', '.loc'))
                                if has_lesion or not lesion_only:
                                    files.append({
                                        'raw_path': raw_path,
                                        'breast_density': self.acrs[bd],
                                        'lesion_density': ld,
                                        'lesion_size': ls,
                                        'has_lesion': has_lesion,
                                        'mask_path': mask_path
                                    })

        self.label_df = pd.DataFrame(files)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        box_series = self.label_df.iloc[idx]
        raw_path = box_series["raw_path"]
        mhd_path = raw_path.replace('.raw', '.mhd')  

        # Read MHD metadata
        im_dims = read_mhd(mhd_path)
        NDims, im_height, im_width = im_dims['NDims'], im_dims['DimSize'][1], im_dims['DimSize'][0]

        # Load image from .raw
        image = np.fromfile(raw_path, dtype="float32").reshape(NDims, im_height, im_width).astype(np.float32)[0]

        # Normalize
        bounds_saved = self.bounds_dict[box_series['breast_density']]
        X = np.std(image[bounds_saved[0], bounds_saved[1]]) * 2
        TH = np.mean(image[bounds_saved[0], bounds_saved[1]])
        image[image < TH - X] = 0
        image[image > TH + X] = X + TH
        image = (image - image.min()) / (image.max() - image.min())
        image = np.rot90(image, k=-1, axes=(-1, -2))
        im_height, im_width = im_width, im_height  # after rot90

        # Get bounding boxes
        boxes, labels = [], []
        loc_path = raw_path.replace('.raw', '.loc')
        if os.path.exists(loc_path):
            x, y, _ = read_loc(loc_path)
            bbox_size = self.bbox_size_dict[box_series["lesion_size"]]
            bbox = get_lesion_box(x, y, image_width=im_width, image_height=im_height, box_side=bbox_size, invert_y=True)
            boxes.append(bbox)
            labels.append(1)
        else:
            boxes = np.zeros((0, 4))

        # Apply window if provided
        if self.window is not None:
            image = window_image(image, *self.window)

        # Convert image to PIL and create dummy mask
        image_pil = Image.fromarray((image * 255).astype(np.uint8)).convert("L")
        mask_dummy = Image.fromarray(np.zeros_like(image, dtype=np.uint8))

        # Resize and pad
        resized_image_pil, _, transform_info = resize_and_pad_pair(
            image_pil, mask_dummy, target_size=(512, 512), fill_image=0, fill_mask=0
        )
        image_tensor = F.to_tensor(resized_image_pil)

        # Rescale bounding boxes
        scale = transform_info['scale']
        pad = transform_info['pad']
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes_rescaled = boxes + np.array([pad[0], pad[1], pad[0], pad[1]])
            boxes_rescaled = boxes_rescaled * scale
        else:
            boxes_rescaled = np.zeros((0, 4))

        # Build target dictionary
        target = {
            'boxes': torch.tensor(boxes_rescaled, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        # # Optional: Apply additional padding or transforms
        if self.padding is not None:
            image_tensor, target = self.padding(image_tensor, target)
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


class EmbedDatasetGenAI_3(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='/projects01/VICTRE/anastasii.sarmakeeva/syntheticDiffusion_breast_data_EMBED/DBT',
                 csv_file='/projects01/didsr-aiml/common_data/EMBED/tables/EMBED_OpenData_metadata_reduced.csv',
                 sample_balance=None,
                 data_split_dict=None,
                 num_positive=None,
                 num_negative=None,
                 one_image_per_patient=False):

        self.root_dir = root_dir
        self.sample_balance = sample_balance
        self.num_positive = num_positive
        self.num_negative = num_negative

        # Load base metadata
        self.label_df = pd.read_csv(os.path.join(self.root_dir, csv_file), low_memory=False)
        self.label_df['anon_dicom_path'] = self.label_df['anon_dicom_path'].str.replace(
            '/mnt/NAS2/mammo/anon_dicom/', f'{self.root_dir}/images/', regex=False)
        self.label_df['anon_dicom_path'] = self.label_df['anon_dicom_path'].apply(lambda p: os.path.dirname(p))

        # Filter only rows with data.npy present
        self.label_df = self.label_df[self.label_df['anon_dicom_path'].apply(
            lambda p: os.path.isfile(os.path.join(p, 'data.npy')))].reset_index(drop=True)

        # Optional: Keep one image per patient
        if one_image_per_patient and 'empi_anon' in self.label_df.columns:
            self.label_df = self.label_df.drop_duplicates(subset=["empi_anon"], keep='first')

        # Filter by sample_balance inside data.npy
        if self.sample_balance is not None:
            keep_indices = []
            for i, row in self.label_df.iterrows():
                npy_path = os.path.join(row['anon_dicom_path'], 'data.npy')
                try:
                    data = np.load(npy_path, allow_pickle=True).item()
                    if data.get('sample_balance') == self.sample_balance:
                        keep_indices.append(i)
                except Exception as e:
                    print(f"Skipping {npy_path}: {e}")
            self.label_df = self.label_df.loc[keep_indices].reset_index(drop=True)

        # Subsample by count if requested
        if self.sample_balance == 'positive' and self.num_positive is not None:
            self.label_df = self.label_df.sample(n=self.num_positive, random_state=42).reset_index(drop=True)
        elif self.sample_balance == 'negative' and self.num_negative is not None:
            self.label_df = self.label_df.sample(n=self.num_negative, random_state=42).reset_index(drop=True)

        print(f"Loaded {len(self.label_df)} samples with sample_balance={self.sample_balance}")

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        entry = self.label_df.iloc[idx]
        npy_path = os.path.join(entry['anon_dicom_path'], 'data.npy')
        data = np.load(npy_path, allow_pickle=True).item()

        image_np = np.array(data['image_diffusion_resized_np'])
        if isinstance(image_np, Image.Image):
            image = F.to_tensor(image_np)
        else:
            if image_np.ndim == 2:
                image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0) / 255.0
            else:
                image = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        box = data.get('box_from_mask')
        if box is not None:
            boxes = torch.tensor([box], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target
