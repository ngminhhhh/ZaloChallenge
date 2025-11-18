from DataPrepare import *
from model.LTS import *
import json

def process_video(frame_encoder, image_encoder, match_head, video_path: str, support_images: List[str]):
    match_head.eval()
    # * Support img embeddings
    support_feats = []
    for img_path in support_images:
        img = Image.open(img_path).convert('RGB')
        emb = image_encoder.extract_features(img, is_normalized=False)
        support_feats.append(emb)

    support_feats = torch.cat(support_feats, dim=0)
    support_embeddings = support_feats.unsqueeze(0)

    cap = cv2.VideoCapture(video_path)

    # * Video property
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    results = []
    frame_idx = 0

    process_bar = tqdm(total=n_frames)

    # * Iterate video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        P3, _, _ = frame_encoder.extract_feature_map(frame)
        P3 = P3.clone()
        heatmap = match_head.forward(P3, support_embeddings)

        print(heatmap)

        frame_idx +=1 
        process_bar.update(1)

    process_bar.close()
    cap.release()

    return results

if __name__ == "__main__":
    dataloader = DataPrepare(root_dir="observing")

    frame_encoder = YOLOv8FeatureExtractor() 
    image_encoder = DINOv2FeatureExtractor()
    match_head = MatchHead()

    train_set = dataloader.prepare_train_data()

    for sample in train_set:
        video_id = sample["video_id"]
        support_images = sample["support_images"]
        query_video = sample["query_video"]

        process_video(frame_encoder, image_encoder, match_head, query_video, support_images)
