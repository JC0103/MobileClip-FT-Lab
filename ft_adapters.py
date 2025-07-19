import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ——— Hyper‑parameters ———————————————————————————————————————————
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32
LR             = 1e-4
WEIGHT_DECAY   = 1e-2
EPOCHS         = 5
TEMPERATURE    = 0.07
BOTTLENECK_DIM = 1024
DATA_ROOT      = "datasets/flickr30k-images"     # contains flickr30k_images/
CAPTION_FILE   = "datasets/results_20130124.token"         # filename<TAB>caption format
SEED           = 42

torch.manual_seed(SEED)
random.seed(SEED)

# ——— 1) Load & freeze MobileCLIP ——————————————————————————————————
base_model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP-S2", pretrained="datacompdr"
)
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
base_model = base_model.to(DEVICE).eval()

# for name, p in base_model.visual.named_parameters():
#     # only unfreeze final_conv, head.fc, and stage 3
#     # if name.startswith("trunk.final_conv") \
#     # or name.startswith("trunk.head.fc") \
#     # or name.startswith("trunk.stages.3"):
#     #     p.requires_grad = True
#     # else:
#         p.requires_grad = False


# determine embedding dim D by a dummy forward
with torch.no_grad():
    dummy = torch.zeros(1,3,224,224, device=DEVICE)
    D = base_model.encode_image(dummy).shape[1]
    print(f"Embedding dimension D: {D}")

# ——— 2) Define and attach adapters ———————————————————————————————
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.ln   = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up   = nn.Linear(bottleneck, dim)
        # start as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self, x):
        return x + self.up(F.relu(self.down(self.ln(x))))

vision_adapter = Adapter(D, BOTTLENECK_DIM).to(DEVICE)
text_adapter   = Adapter(D, BOTTLENECK_DIM).to(DEVICE)

class ClipWithAdapters(nn.Module):
    def __init__(self, base, va, ta):
        super().__init__()
        self.base = base
        self.va   = va
        self.ta   = ta
    def encode_image(self, x):
        return self.va(self.base.encode_image(x))
    def encode_text(self, x):
        return self.ta(self.base.encode_text(x))

model = ClipWithAdapters(base_model, vision_adapter, text_adapter)

# ——— 3) Prepare Flickr30k + train/val split —————————————————————————
full_ds = Flickr30k(root=DATA_ROOT, ann_file=CAPTION_FILE, transform=preprocess)
n = len(full_ds)
n_train = int(0.8*n)
train_base, val_base = random_split(full_ds, [n_train, n-n_train],
                                    generator=torch.Generator().manual_seed(SEED))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, tokenizer, seed=0):
        self.base_ds = base_ds
        self.tok     = tokenizer
        self.rng     = random.Random(seed)
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, caps = self.base_ds[idx]
        cap = self.rng.choice(caps)
        tok = self.tok([cap]).squeeze(0)
        return img, tok

train_ds = PairDataset(train_base, tokenizer, seed=SEED)
val_ds   = PairDataset(val_base,   tokenizer, seed=SEED)

def collate_fn(batch):
    imgs, toks = zip(*batch)
    imgs = torch.stack(imgs).to(DEVICE)
    toks = torch.stack(toks).to(DEVICE)
    return imgs, toks

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

# ——— 4) Optimizer (only adapters) —————————————————————————————————
optimizer = torch.optim.AdamW(
    list(vision_adapter.parameters()) + list(text_adapter.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)

# ——— 5) Train + validate + record metrics ————————————————————————
train_losses, val_losses = [], []
r1_list, r5_list, r10_list = [], [], []

def recall_at_k(image_embs, text_embs, cpi, K, chunk_size=512):
    N, D = image_embs.shape
    hits = 0
    for start in range(0, N, chunk_size):
        end = min(N, start+chunk_size)
        chunk = image_embs[start:end]             # (C, D)
        sim = chunk @ text_embs.T                 # (C, 5N)
        sim = sim / (chunk.norm(2,1,True) * text_embs.norm(2,1,True).T)
        topk = sim.topk(K,1).indices              # (C,K)
        for i, row in enumerate(topk):
            idx = start + i
            gt = set(range(idx*cpi, idx*cpi+cpi))
            if gt & set(row.tolist()):
                hits += 1
    return hits / N

for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    total_loss = 0.0
    bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")
    for imgs, toks in bar:
        optimizer.zero_grad()
        ie = model.encode_image(imgs)    # (B,D)
        te = model.encode_text(toks)     # (B,D)
        logits = ie @ te.T / TEMPERATURE
        labels = torch.arange(len(imgs), device=DEVICE)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*imgs.size(0)
        bar.set_postfix(train_loss=f"{loss.item():.4f}")
    train_loss = total_loss/len(train_loader.dataset)
    train_losses.append(train_loss)

    # validate
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        bar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch}/{EPOCHS}")
        for imgs, toks in bar:
            ie = model.encode_image(imgs)
            te = model.encode_text(toks)
            logits = ie @ te.T / TEMPERATURE
            labels = torch.arange(len(imgs), device=DEVICE)
            batch_loss = F.cross_entropy(logits, labels).item()
            total_loss += batch_loss*imgs.size(0)
            bar.set_postfix(val_loss=f"{batch_loss:.4f}")
    val_loss = total_loss/len(val_loader.dataset)
    val_losses.append(val_loss)

    # retrieval on full val split (5 captions each)
    ret_loader = DataLoader(
        val_base, batch_size=32, shuffle=False, num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([img for img,_ in batch]).to(DEVICE),
            [caps for _,caps in batch]
        )
    )
    all_ie, all_te = [], []
    with torch.no_grad():
        bar = tqdm(ret_loader, desc=f"[Ret]   Epoch {epoch}/{EPOCHS}")
        for imgs, cap_lists in bar:
            ie = model.encode_image(imgs)
            flat_caps = [c for caps in cap_lists for c in caps]
            toks = tokenizer(flat_caps).to(DEVICE)
            te = model.encode_text(toks)
            all_ie.append(ie.cpu())
            all_te.append(te.cpu())
    all_ie = torch.cat(all_ie, dim=0)
    all_te = torch.cat(all_te, dim=0)

    r1  = recall_at_k(all_ie, all_te, 5, 1)
    r5  = recall_at_k(all_ie, all_te, 5, 5)
    r10 = recall_at_k(all_ie, all_te, 5, 10)
    r1_list.append(r1); r5_list.append(r5); r10_list.append(r10)

    print(f"Epoch {epoch}: Train {train_loss:.4f}  Val {val_loss:.4f}  R@1 {r1:.3f} R@5 {r5:.3f} R@10 {r10:.3f}")

# ——— 6) Plot Loss & Recall Curves ——————————————————————————————————
epochs = list(range(1, EPOCHS+1))

plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses,   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('FT‑Adapter: Loss vs. Epoch')
plt.legend()

plt.figure()
plt.plot(epochs, r1_list,  marker='o', label='R@1')
plt.plot(epochs, r5_list,  marker='o', label='R@5')
plt.plot(epochs, r10_list, marker='o', label='R@10')
plt.xlabel('Epoch'); plt.ylabel('Recall')
plt.title('FT‑Adapter: Recall@K vs. Epoch')
plt.ylim(0,1.0)
plt.legend()
plt.show()
