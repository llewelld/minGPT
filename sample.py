from pytorch_lightning import seed_everything
import torch
from mingpt.utils import sample
from mingpt.model import GPT

if __name__ == '__main__':
    seed_everything(42)

    checkpoint='trained.ckpt'
    model = GPT.load_from_checkpoint(checkpoint)
    model.eval()

    context = "By artificial intelligence, or wrong surmise,"
    x = torch.tensor([model.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
    y = sample(model, x, 1000, temperature=0.9, sample=True, top_k=5)[0]
    completion = ''.join([model.itos[int(i)] for i in y])
    print(completion)

