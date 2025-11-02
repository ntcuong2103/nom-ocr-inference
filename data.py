from nom_ids_ocr.data import SeqVocab
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import image
from PIL import Image 
from pathlib import Path
import imagesize

class ImageDatasetBBox(Dataset):
    def __init__(self, image_dir, label_dir, vocab, transform=None, expand_ratio=1.2):
        self.transform = transform
        self.vocab = vocab
        self.image_dir = Path(image_dir) # Convert to Path object
        self.label_dir = Path(label_dir) # Convert to Path object
        self.image_paths = []
        self.label_paths = []
        self.expand_ratio = expand_ratio
        self.load_data()
        self.bboxes_info = []  # To store (image_path, bbox, class) tuples
        self.prepare_bboxes()

    def load_data(self):
        # Clear previous lists in case load_data is called multiple times
        self.image_paths = []
        self.label_paths = []

        for image_path in self.image_dir.rglob('*'): # Use rglob for recursive search
            if image_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                # Construct corresponding label path
                relative_path = image_path.relative_to(self.image_dir)
                label_path = (self.label_dir / relative_path).with_suffix('.txt')

                # Check if label file exists
                if label_path.exists():
                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)
                else:
                    print(f"Label file not found for {image_path}. Skipping.")

    def prepare_bboxes(self):
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            width, height = imagesize.get(image_path)
            
            with open(label_path, 'r') as f:
                labels = f.readlines()
            classes = [label.strip().split()[0] for label in labels]
            bboxes = [list(map(float, label.strip().split()[1:])) for label in labels]
            classes = np.array(classes)
            bboxes = np.array(bboxes)
            bboxes = bboxes * np.array([width, height, width, height])
            bboxes = bboxes.astype(int)

            for bbox, cls in zip(bboxes, classes):
                if cls in self.vocab.ids_dict or cls == '0':
                    self.bboxes_info.append((image_path, bbox, cls))

    def __len__(self):
        return len(self.bboxes_info)

    def __getitem__(self, idx):
        image_path, bbox, cls = self.bboxes_info[idx]
        image = Image.open(image_path).convert('RGB')
        x, y, w_bbox, h_bbox = bbox
        
        # make equal crop
        w = max(w_bbox, h_bbox) * self.expand_ratio
        h = w

        # calculate coordinates
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(image.width, int(x + w / 2))
        y2 = min(image.height, int(y + h / 2))
        # crop the image
        image_cropped = image.crop((x1, y1, x2, y2))
        # check size of the cropped image
        if image_cropped.size[0] < 1 or image_cropped.size[1] < 1:
            print(f"Invalid crop size for {image_path} at index {idx}. Skipping.")
            return None
        if self.transform:
            image_cropped = self.transform(image_cropped)
        return f'{image_path}@{idx}', image_cropped, self.vocab.encode(cls)

if __name__ == "__main__":
    base_vocab = open('vocab_ids.txt', 'r').read().split('\n')
    ids_dict = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('ids_exp.txt', 'r').readlines()}
    eval_transforms = transforms.Compose([
            transforms.Resize(size=128),
            transforms.RandomCrop(size=128),
            transforms.RandomInvert(p=1.0),
        ])

    dataset = ImageDatasetBBox(
        image_dir='datasets/nomnaocr/images',
        label_dir='detection/nomnaocr/labels',
        vocab=SeqVocab(base_vocab, ids_dict),
        transform=eval_transforms,
        expand_ratio=1.2
    )

    print(f"Dataset size: {len(dataset)}")


