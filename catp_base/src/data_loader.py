import os
import platform
from collections import defaultdict
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore

from src.types import SampleID, TextContent


class TaskDataset(Dataset):
    """
    A PyTorch-style dataset that loads image and text data for a specific task.
    Each sample is identified by an integer ID, and can have
    multiple data types (images, texts) stored in dictionaries.
    """

    input_data: Dict[SampleID, Dict[str, torch.Tensor | TextContent]]
    output_data: Dict[SampleID, Dict[str, torch.Tensor | TextContent]]
    sample_ids: List[SampleID]

    def __init__(self, data_path: str, *, task_id: int):
        """
        Initialize dataset with the path to the data and a task ID.

        :param data_path: Path to the root data directory.
        :param task_id: An integer identifying which task's data to load.
        """
        # For example: data_path/task_id
        self.task_id = task_id
        self.task_data_path = os.path.join(data_path, str(task_id))
        self._load_data()

        # Store and sort all sample IDs (assumes input_data covers all samples)
        self.sample_ids = sorted(self.input_data.keys())

    @staticmethod
    def _load_images(image_dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor]]:
        """
        Load images from a directory, storing each image in a tensor.
        The dictionary key is derived from the file name (without extension).
        """
        result = {}
        image_processor = transforms.Compose([
            transforms.PILToTensor(),
        ])
        for image_filename in os.listdir(image_dir_path):
            # Extract numeric ID from file name, e.g. '5.jpg' -> sample_id=5
            sample_id = int(os.path.splitext(image_filename)[0])
            image_full_path = os.path.join(image_dir_path, image_filename)
            with Image.open(image_full_path) as image:
                # Convert to RGB and apply transforms
                result[sample_id] = {
                    'image': image_processor(image.convert("RGB"))
                }
        return result

    @staticmethod
    def _load_text(file_path: str, sequential=True) -> Dict[SampleID, Dict[str, TextContent]]:
        """
        Load text lines from a .txt file.
        Each line is mapped to a sample ID based on line-order (0-based).
        """
        # for non-sequential tasks, there may exist multiple textual ground truths.
        # so we need to distingush them.
        if sequential:
            key_postfix = ''
        else:
            if platform.system() == 'Windows':
                key_postfix = '-' + file_path.split('\\')[-1][:-4].split('-')[0]
            else:
                key_postfix = '-' + file_path.split('/')[-1][:-4].split('-')[0]
        result = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                sample_id: SampleID = idx
                result[sample_id] = {
                    'text' + key_postfix: line.strip()
                }
        return result

    def _load_files(self, dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor | TextContent]]:
        """
        Load all files under a given directory, merging data for each sample ID.
        Recognizes 'images' directory and .txt files, raises ValueError otherwise.
        """
        result = defaultdict(dict)
        for item in os.listdir(dir_path):
            item_full_path = os.path.join(dir_path, item)

            if item == 'images':
                file_data = self._load_images(item_full_path)
            elif item.endswith('.txt'):
                file_data = self._load_text(item_full_path, sequential=self.task_id < 200)
            else:
                raise ValueError(f"Unknown file type or directory: {item_full_path}")

            for sample_id, data_dict in file_data.items():
                result[sample_id].update(data_dict)

        return dict(result)

    def _load_data(self) -> None:
        input_path = os.path.join(self.task_data_path, 'inputs')
        output_path = os.path.join(self.task_data_path, 'outputs')

        self.input_data = self._load_files(input_path)
        self.output_data = self._load_files(output_path)

    def __getitem__(self, index: int) -> Dict[str, int | Dict[str, torch.Tensor | TextContent]]:
        sample_id = self.sample_ids[index]
        data_item = {
            'sample_id': sample_id,
            'input': self.input_data[sample_id],
            'output': self.output_data[sample_id]
        }
        return data_item

    def __len__(self) -> int:
        return len(self.sample_ids)
