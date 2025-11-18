import json 
from pathlib import Path
from typing import List, Dict

class DataPrepare:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / 'train'
        self.test_dir = self.root_dir / 'public_test'
        self.annotations_file = self.train_dir / 'annotations' / 'annotations.json'
        
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def get_annotations_for_sample(self, folder_name: str) -> Dict:
        for annotation in self.annotations:
            if annotation.get('video_id') == folder_name:
                return annotation

    def get_class_folders(self, folder_dir) -> List[str]:
        class_folder_dir = folder_dir / 'samples'
        class_folders = []
        for item in class_folder_dir.iterdir():
            if item.is_dir():
                class_folders.append(item.name)

        return class_folders
    
    def get_support_image(self, folder_dir, class_folder) -> List[str]:
        object_images_dir = folder_dir / 'samples' / class_folder / 'object_images'
        image_files = [f for f in object_images_dir.iterdir()]

        return [str(img) for img in image_files]
    
    def get_query_video(self, folder_dir, class_folder) -> str:
        video_path = folder_dir / 'samples' / class_folder / "drone_video.mp4"
        return str(video_path)
    
    def prepare_train_data(self) -> List[Dict]:
        class_folders = self.get_class_folders(self.train_dir)
        train_data = []

        for folder in class_folders:
            support_images = self.get_support_image(self.train_dir, folder)
            query_video = self.get_query_video(self.train_dir, folder)
            sample = {
                'video_id': folder,
                'support_images': support_images,
                'num_support_images': len(support_images),
                'query_video': query_video
            }

            train_data.append(sample)

        return train_data
    
    def prepare_test_data(self) -> List[Dict]:
        class_folders = self.get_class_folders(self.test_dir)
        test_data = []

        for folder in class_folders:
            support_images = self.get_support_image(folder, self.train_dir)
            query_video = self.get_query_video(folder, self.train_dir)
            sample = {
                'folder': folder,
                'support_images': support_images,
                'num_support_images': len(support_images),
                'query_video': query_video
            }

            test_data.append(sample)

        return test_data
    

    
