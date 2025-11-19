from model.LTS import *
from DataPrepare import DataPrepare
from tqdm import tqdm
import cv2

def process_video(video_id: str, video_path: str, frame_encoder, query_embs: torch.Tensor):
    cap = cv2.VideoCapture(video_path)

    # * video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    results = []
    frame_idx = 0

    # * Process bar 
    process_bar = tqdm(total=n_frames)

    # * Iterate video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        P3, P4, P5 = frame_encoder.extract_feature_map(frame)
        P3, P4, P5 = P3.clone(), P4.clone(), P5.clone()

        feature_map = frame_encoder.fusion_feature_map(P3, P4, P5)

        print(f"Feature_maps shape: {feature_map.shape}")

        frame_idx += 1
        process_bar.update(1)

    process_bar.close()
    cap.release()

    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataPrepare(root_dir="observing")
    train_set = dataloader.prepare_train_data()

    image_encoder = CLIPEncoder(device=device)
    frame_encoder = YOLOv8mEncoder(device=device)

    for sample in train_set:
        video_id = sample["video_id"]
        query_images = sample["query_images"]
        query_video = sample["query_video"]

        # * Query embedding
        query_embs = []
        for img_path in query_images:
            query_embs.append(image_encoder.extract_features(image_path=img_path))
        query_embs = torch.cat(query_embs, dim=0).unsqueeze(0)
        
        process_video(video_id=video_id, video_path=query_video, frame_encoder=frame_encoder, query_embs=query_embs)

        break


