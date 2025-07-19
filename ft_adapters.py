import random, os
import torch, open_clip
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

# ————————————————————————————————————————————————————————————————
# 1) GLOBAL SETUP
# ————————————————————————————————————————————————————————————————
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 32
WEIGHT_DECAY = 1e-2
EPOCHS       = 5
TEMPERATURE  = 0.07
SEED         = 42
DATA_ROOT    = "datasets/flickr30k-images"
CAP_FILE     = "datasets/results_20130124.token"
os.makedirs("plots", exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)

# 1a) load & freeze backbone
backbone, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP-S2", pretrained="datacompdr"
)
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters(): p.requires_grad = False

# 1b) get embedding dim
with torch.no_grad():
    D = backbone.encode_image(torch.zeros(1,3,224,224, device=DEVICE)).shape[1]

# 1c) build dataset & splits
full = Flickr30k(root=DATA_ROOT, ann_file=CAP_FILE, transform=preprocess)
n_train = int(0.8*len(full))
train_base, val_base = random_split(full, [n_train, len(full)-n_train],
                                    generator=torch.Generator().manual_seed(SEED))

class PairDS(torch.utils.data.Dataset):
    def __init__(self, base, tok, seed=0):
        self.base, self.tok, self.rng = base, tok, random.Random(seed)
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        img,caps = self.base[i]; cap=self.rng.choice(caps)
        return img, self.tok([cap]).squeeze(0)

def collate_pair(batch):
    imgs, toks = zip(*batch)
    return torch.stack(imgs,dim=0).to(DEVICE), torch.stack(toks,dim=0).to(DEVICE)

train_ds = PairDS(train_base, tokenizer, seed=SEED)
val_ds   = PairDS(val_base,   tokenizer, seed=SEED)
train_loader = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_pair, num_workers=0)
val_loader   = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_pair, num_workers=0)

def recall_at_k(img_embs, text_embs, cpi, K, chunk=512):
    N = img_embs.size(0)
    hits=0
    for i in range(0,N,chunk):
        chunk_emb = img_embs[i:i+chunk]                      # (C,D)
        sim = chunk_emb @ text_embs.T                       # (C,5N)
        sim = sim/(chunk_emb.norm(2,1,True)*text_embs.norm(2,1,True).T)
        topk = sim.topk(K,dim=1).indices                     # (C,K)
        for j,row in enumerate(topk):
            idx = i+j
            gt = set(range(idx*cpi, idx*cpi+cpi))
            if gt & set(row.tolist()): hits+=1
    return hits/N

# ————————————————————————————————————————————————————————————————
# 2) ADAPTER + TRAINING FUNCTION
# ————————————————————————————————————————————————————————————————
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim)
        )
        # init as identity
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return x + self.net(x)

class ClipWithAdapters(nn.Module):
    def __init__(self, bb, va, ta):
        super().__init__()
        self.bb, self.va, self.ta = bb, va, ta
        self.bb.eval()
    def encode_image(self,x):
        with torch.no_grad(): emb = self.bb.encode_image(x)
        return self.va(emb)
    def encode_text(self,x):
        with torch.no_grad(): emb = self.bb.encode_text(x)
        return self.ta(emb)

def train_and_eval(bottleneck, lr):
    # build model & optimizer
    va = Adapter(D,bottleneck).to(DEVICE)
    ta = Adapter(D,bottleneck).to(DEVICE)
    model = ClipWithAdapters(backbone,va,ta).to(DEVICE)
    optim = torch.optim.AdamW(list(va.parameters())+list(ta.parameters()),
                              lr=lr, weight_decay=WEIGHT_DECAY)

    train_losses, val_losses = [], []
    r1s,r5s,r10s = [],[],[]

    for ep in range(1,EPOCHS+1):
        # train
        model.train(); tot=0
        for imgs,toks in tqdm(train_loader,desc=f"[B={bottleneck} LR={lr}] Train ep{ep}"):
            optim.zero_grad()
            ie,te = model.encode_image(imgs), model.encode_text(toks)
            logits = ie@te.T/TEMPERATURE
            labels = torch.arange(len(imgs),device=DEVICE)
            loss   = F.cross_entropy(logits,labels)
            loss.backward(); optim.step()
            tot += loss.item()*imgs.size(0)
        train_losses.append(tot/len(train_loader.dataset))

        # val
        model.eval(); tot=0
        with torch.no_grad():
            for imgs,toks in tqdm(val_loader, desc=f"[B={bottleneck} LR={lr}] Val ep{ep}"):
                ie,te = model.encode_image(imgs), model.encode_text(toks)
                logits = ie@te.T/TEMPERATURE
                labels = torch.arange(len(imgs),device=DEVICE)
                tot += F.cross_entropy(logits,labels,reduction='sum').item()
        val_losses.append(tot/len(val_loader.dataset))

        # full‐split recall
        all_ie, all_te = [],[]
        raw_loader = DataLoader(val_base,batch_size=BATCH_SIZE,shuffle=False, num_workers=0,
            collate_fn=lambda b:(torch.stack([img for img,_ in b]).to(DEVICE), [caps for _,caps in b]))
        with torch.no_grad():
            for imgs,caps in tqdm(raw_loader, desc=f"[B={bottleneck} LR={lr}] Ret ep{ep}"):
                ie = model.encode_image(imgs)
                flat = [c for cl in caps for c in cl]
                te = model.encode_text(tokenizer(flat).to(DEVICE))
                all_ie.append(ie.cpu()); all_te.append(te.cpu())
        all_ie = torch.cat(all_ie); all_te=torch.cat(all_te)

        r1s.append( recall_at_k(all_ie,all_te,5, 1) )
        r5s.append( recall_at_k(all_ie,all_te,5, 5) )
        r10s.append(recall_at_k(all_ie,all_te,5,10) )

    # save plots
    epochs = list(range(1,EPOCHS+1))
    # learning curve
    plt.figure(); plt.plot(epochs,train_losses,label='train'); plt.plot(epochs,val_losses,label='val')
    plt.title(f"Loss B={bottleneck} LR={lr}"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(f"plots/loss_b{bottleneck}_lr{lr}.png"); plt.close()
    # recall curves
    plt.figure()
    plt.plot(epochs,r1s,marker='o',label='R@1')
    plt.plot(epochs,r5s,marker='o',label='R@5')
    plt.plot(epochs,r10s,marker='o',label='R@10')
    plt.title(f"Recall B={bottleneck} LR={lr}"); plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.ylim(0,1); plt.legend()
    plt.savefig(f"plots/recall_b{bottleneck}_lr{lr}.png"); plt.close()

    return {"train":train_losses, "val":val_losses, "r1":r1s, "r5":r5s, "r10":r10s}

# ————————————————————————————————————————————————————————————————
# 3) SWEEP OVER BOTTLENECK SIZES
# ————————————————————————————————————————————————————————————————
b_sizes = [256,512,1024]
metrics_b = {}
for b in b_sizes:
    metrics_b[b] = train_and_eval(b, lr=1e-4)

# bar‐chart of *final* recall@K
final = {b:(metrics_b[b]["r1"][-1], metrics_b[b]["r5"][-1], metrics_b[b]["r10"][-1]) for b in b_sizes}
import numpy as np
x = np.arange(len(b_sizes))
width=0.25
fig,ax=plt.subplots()
for i,k in enumerate(["r1","r5","r10"]):
    vals = [ final[b][i] for b in b_sizes ]
    ax.bar(x + (i-1)*width, vals, width, label=f"R@{ [1,5,10][i] }")
ax.set_xticks(x); ax.set_xticklabels(b_sizes)
ax.set_xlabel("Bottleneck"); ax.set_ylabel("Recall"); ax.set_title("Final Recall vs Bottleneck")
ax.legend(); fig.savefig("plots/recall_bar_bottleneck.png"); plt.close()

# pick best bottleneck by highest R@10
best_b = max(b_sizes, key=lambda b: final[b][2])
print("→ best bottleneck =", best_b)

# ————————————————————————————————————————————————————————————————
# 4) SWEEP OVER LEARNING RATES FOR best_b
# ————————————————————————————————————————————————————————————————
lrs = [1e-4, 5e-4, 1e-3, 5e-3]
metrics_lr={}
for lr in lrs:
    metrics_lr[lr] = train_and_eval(best_b, lr)

# bar‐chart of *final* recall@K across lrs
final_lr = {lr:(metrics_lr[lr]["r1"][-1],metrics_lr[lr]["r5"][-1],metrics_lr[lr]["r10"][-1]) for lr in lrs}
x = np.arange(len(lrs)); width=0.2
fig,ax=plt.subplots()
for i,k in enumerate(["r1","r5","r10"]):
    vals = [ final_lr[lr][i] for lr in lrs ]
    ax.bar(x + (i-1)*width, vals, width, label=f"R@{ [1,5,10][i] }")
ax.set_xticks(x); ax.set_xticklabels(lrs)
ax.set_xlabel("Learning Rate"); ax.set_ylabel("Recall"); ax.set_title(f"Final Recall vs LR @ B={best_b}")
ax.legend(); fig.savefig("plots/recall_bar_lr.png"); plt.close()
