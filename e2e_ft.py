import matplotlib.pyplot as plt
import random
import torch
import open_clip
import torch.nn.functional as F
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

# ——— Hyper‑parameters —————————————————————————————————————————
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32
LR             = 1e-5
WEIGHT_DECAY   = 1e-2
EPOCHS         = 5
TEMPERATURE    = 0.07
DATA_ROOT      = "datasets/flickr30k-images"     # contains flickr30k_images/
CAPTION_FILE   = "datasets/results_20130124.token"         # filename<TAB>caption format
SEED           = 42

# fix randomness
torch.manual_seed(SEED)
random.seed(SEED)

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

# ——— 1) Load MobileCLIP ——————————————————————————————————————
model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP-S2", pretrained="datacompdr"
)
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
model = model.to(DEVICE)

# ——— 2) Load + split Flickr30k ——————————————————————————————
full_ds = Flickr30k(
    root=DATA_ROOT,
    ann_file=CAPTION_FILE,
    transform=preprocess
)
n = len(full_ds)
n_train = int(0.8 * n)
n_val   = n - n_train
train_base, val_base = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

# ——— 3) Build a pair‑dataset (one caption per image) —————————————————
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, tokenizer, seed=0):
        self.base_ds = base_ds
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, caps = self.base_ds[idx]
        cap = self.rng.choice(caps)              # pick one of 5
        tok = self.tokenizer([cap]).squeeze(0)   # (seq_len,)
        return img, tok

train_ds = PairDataset(train_base, tokenizer, seed=SEED)
val_ds   = PairDataset(val_base,   tokenizer, seed=SEED)

def collate_fn(batch):
    imgs, toks = zip(*batch)
    imgs = torch.stack(imgs, dim=0).to(DEVICE)   # (B,3,H,W)
    toks = torch.stack(toks, dim=0).to(DEVICE)   # (B,seq_len)
    return imgs, toks

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

# ——— 4) Optimizer ———————————————————————————————————————————
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ——— 5) Train + validate —————————————————————————————————————
train_losses = []
val_losses   = []
r1_list, r5_list, r10_list = [], [], []

for epoch in range(1, EPOCHS+1):
    # —— training with progress bar
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
    for images, tokens in train_bar:
        optimizer.zero_grad()
        img_emb = model.encode_image(images)   # (B, D)
        txt_emb = model.encode_text(tokens)    # (B, D)
        logits  = img_emb @ txt_emb.t() / TEMPERATURE
        labels  = torch.arange(len(images), device=DEVICE)
        loss    = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        train_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # —— validation with progress bar
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for images, tokens in val_bar:
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(tokens)
            logits  = img_emb @ txt_emb.t() / TEMPERATURE
            labels  = torch.arange(len(images), device=DEVICE)
            batch_loss = F.cross_entropy(logits, labels).item()
            val_loss   += batch_loss * images.size(0)
            val_bar.set_postfix({'batch_loss': f"{batch_loss:.4f}"})
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # —— retrieval with progress bar
    val_ret_loader = DataLoader(
        val_base,               # already has transform=preprocess
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([img for img, _ in batch], dim=0).to(DEVICE),
            [caps for _, caps in batch]
        )
    )
    all_image_embs, all_text_embs = [], []
    ret_bar = tqdm(val_ret_loader, desc=f"Epoch {epoch}/{EPOCHS} [Ret]", leave=False)
    with torch.no_grad():
        for images, cap_lists in ret_bar:
            img_feats = model.encode_image(images)  # (B, D)
            flat_caps = [c for caps in cap_lists for c in caps]
            toks      = tokenizer(flat_caps).to(DEVICE)
            txt_feats = model.encode_text(toks)     # (B*5, D)
            all_image_embs.append(img_feats)
            all_text_embs.append(txt_feats)
    all_image_embs = torch.cat(all_image_embs, dim=0).cpu()
    all_text_embs  = torch.cat(all_text_embs,  dim=0).cpu()

    r1  = recall_at_k(all_image_embs, all_text_embs, 5, 1,  chunk_size=512)
    r5  = recall_at_k(all_image_embs, all_text_embs, 5, 5,  chunk_size=512)
    r10 = recall_at_k(all_image_embs, all_text_embs, 5, 10, chunk_size=512)
    r1_list.append(r1); r5_list.append(r5); r10_list.append(r10)

    print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}")

# ——— 6) Plot ——————————————————————————————————————————————
epochs = list(range(1, EPOCHS + 1))

plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# plot retrieval metrics
plt.figure()
plt.plot(epochs, r1_list,  marker='o', label='Recall@1')
plt.plot(epochs, r5_list,  marker='o', label='Recall@5')
plt.plot(epochs, r10_list, marker='o', label='Recall@10')
plt.xlabel('Epoch'); plt.ylabel('Recall')
plt.title('Validation Recall@K')
plt.ylim(0, 1.0)
plt.legend()
plt.show()