import open_clip
import torch
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 1) Load model, image‑preprocess, and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP-S2', pretrained='datacompdr'
)
tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
model = model.to(device).eval()

# 2) Point to your local data
images_root = "datasets/flickr30k-images"    #contains flickr30k_images/
captions_tsv ="datasets/results_20130124.token"      # your “filename<TAB>caption” file

# 3) Create the dataset (returns (PIL.Image, list-of-5-captions))
dataset = Flickr30k(
    root=images_root,
    ann_file=captions_tsv,
    transform=preprocess,        # uses the exact MobileCLIP preprocessing
)

# 4) Collate & tokenize
def collate_fn(batch):
    images, cap_lists = zip(*batch)            # batch_size tuples
    images = torch.stack(images, dim=0).to(device)
    # flatten captions
    flat_caps = [c for caps in cap_lists for c in caps]
    # tokenize into a tensor of shape (batch_size*5, seq_len)
    text_tokens = tokenizer(flat_caps).to(device)
    return images, text_tokens, cap_lists

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# 5) Zero‑shot inference loop
import torch.nn.functional as F

all_image_embs = []
all_text_embs  = []

print("Starting zero-shot inference..., data size:", len(dataset))

with torch.no_grad():
    for images, text_tokens, _ in tqdm(loader, desc="Zero‑shot inference", total=len(loader)):
        img_feats = model.encode_image(images)      # (B, dim)
        txt_feats = model.encode_text(text_tokens)   # (B*5, dim)
        all_image_embs.append(img_feats)
        all_text_embs.append(txt_feats)


all_image_embs = torch.cat(all_image_embs, dim=0)  # (N, dim)
all_text_embs  = torch.cat(all_text_embs,  dim=0)  # (N*5, dim)

# 6) Quick sanity‑check prints
print(f"Image embeds: {all_image_embs.shape}")
print(f"Text  embeds: {all_text_embs.shape}")

# 7) Compute cosine similarities
#    (N×D) @ (D×(5N)) -> (N,5N)
# sim = all_image_embs @ all_text_embs.T
# sim = sim / (
#     all_image_embs.norm(dim=1, keepdim=True) 
#     * all_text_embs.norm(dim=1, keepdim=True).T
# )

# 8) Build a helper to compute Recall@K
def recall_at_k(image_embs, text_embs, captions_per_image, K, chunk_size=1024):
    """
    Compute Recall@K in chunks to avoid OOM on GPU.
    image_embs: Tensor[N, D]
    text_embs:  Tensor[N*captions_per_image, D]
    """
    N, D = image_embs.shape
    M, D2 = text_embs.shape
    assert D == D2 and M == N * captions_per_image
    hits = 0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        emb_chunk = image_embs[start:end]             # (C, D)
        # similarity (C, M)
        sim = emb_chunk @ text_embs.T
        # normalize
        sim = sim / (
            emb_chunk.norm(dim=1, keepdim=True) 
            * text_embs.norm(dim=1, keepdim=True).T
        )
        # top-K indices for this chunk
        topk = sim.topk(K, dim=1).indices            # (C, K)
        for i, row in enumerate(topk):
            idx = start + i
            gt = set(range(idx*captions_per_image,
                           idx*captions_per_image+captions_per_image))
            if gt & set(row.tolist()):
                hits += 1
    return hits / N


# 9) Compute & print Recall@1,5,10
r1  = recall_at_k(all_image_embs, all_text_embs, captions_per_image=5, K=1,  chunk_size=1024)
r5  = recall_at_k(all_image_embs, all_text_embs, captions_per_image=5, K=5,  chunk_size=1024)
r10 = recall_at_k(all_image_embs, all_text_embs, captions_per_image=5, K=10, chunk_size=1024)
print(f"R@1: {r1:.3f}, R@5: {r5:.3f}, R@10: {r10:.3f}")

