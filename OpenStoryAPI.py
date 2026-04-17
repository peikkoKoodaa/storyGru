import os
import re
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# ASETUKSET
# -----------------------------
MODEL_FILE = "story_gru_model.pth"
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_LENGTH = 250
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_REPETITION_WINDOW = 80

device = torch.device("cpu")

# -----------------------------
# MALLI
# -----------------------------
class ManualGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.xz = nn.Linear(input_size, hidden_size)
        self.hz = nn.Linear(hidden_size, hidden_size, bias=False)

        self.xr = nn.Linear(input_size, hidden_size)
        self.hr = nn.Linear(hidden_size, hidden_size, bias=False)

        self.xn = nn.Linear(input_size, hidden_size)
        self.hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.xz(x) + self.hz(h))
        r = torch.sigmoid(self.xr(x) + self.hr(h))
        n = torch.tanh(self.xn(x) + self.hn(r * h))
        h_new = (1 - z) * n + z * h
        return h_new


class StoryGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            inp = embed_size if i == 0 else hidden_size
            self.layers.append(ManualGRULayer(inp, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x)

        if h is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]

        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]

            new_h = []
            for layer_idx, layer in enumerate(self.layers):
                h_t = layer(inp, h[layer_idx])
                inp = self.dropout(h_t) if layer_idx < self.num_layers - 1 else h_t
                new_h.append(h_t)

            h = new_h
            outputs.append(inp.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        out = self.fc(out)
        return out, h


# -----------------------------
# APUTOIMINNOT
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    recent_tokens=None,
    repetition_penalty=1.0
):
    logits = logits.clone()

    if recent_tokens is not None and repetition_penalty > 1.0:
        for token in set(recent_tokens):
            if logits[token] > 0:
                logits[token] /= repetition_penalty
            else:
                logits[token] *= repetition_penalty

    logits = logits / max(temperature, 1e-6)

    if top_k is not None and 0 < top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits[indices] = values
        logits = filtered_logits

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


@torch.no_grad()
def generate_text(
    model,
    prompt,
    stoi,
    itos,
    length=250,
    temperature=0.7,
    top_k=5,
    repetition_penalty=1.15,
    repetition_window=80
):
    model.eval()

    prompt = clean_text(prompt)
    filtered_prompt = "".join(ch for ch in prompt if ch in stoi)
    if not filtered_prompt:
        filtered_prompt = DEFAULT_PROMPT

    input_tokens = [stoi[ch] for ch in filtered_prompt]
    x = torch.tensor([input_tokens], dtype=torch.long).to(device)

    h = None
    _, h = model(x, h)

    generated = filtered_prompt
    recent_tokens = input_tokens.copy()
    last_token = x[:, -1:]

    for _ in range(length):
        out, h = model(last_token, h)
        logits = out[:, -1, :].squeeze(0)

        next_token = sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            recent_tokens=recent_tokens[-repetition_window:],
            repetition_penalty=repetition_penalty
        )

        generated += itos[next_token]
        recent_tokens.append(next_token)
        last_token = torch.tensor([[next_token]], dtype=torch.long).to(device)

        if generated.count("Once upon a time") > 1:
            break

    return clean_text(generated)


def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpointia ei löytynyt: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Story Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
stoi = None
itos = None

class GenerateRequest(BaseModel):
    prompt: str
    length: int = DEFAULT_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


@app.on_event("startup")
def startup_event():
    global model, stoi, itos

    checkpoint = load_checkpoint(MODEL_FILE)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    cfg = checkpoint.get("config", {})

    vocab_size = len(stoi)

    model_instance = StoryGRU(
        vocab_size=vocab_size,
        embed_size=cfg.get("EMBED_SIZE", 64),
        hidden_size=cfg.get("HIDDEN_SIZE", 128),
        num_layers=cfg.get("NUM_LAYERS", 2),
        dropout=cfg.get("DROPOUT", 0.1)
    ).to(device)

    model_instance.load_state_dict(checkpoint["model_state"])
    model_instance.eval()

    model = model_instance
    print("Malli ladattu onnistuneesti.")


@app.get("/")
def root():
    return {"message": "Story Generator API is running"}


@app.post("/generate")
def generate(req: GenerateRequest):
    story = generate_text(
        model=model,
        prompt=req.prompt,
        stoi=stoi,
        itos=itos,
        length=req.length,
        temperature=req.temperature,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        repetition_window=DEFAULT_REPETITION_WINDOW
    )

    return {
        "prompt": req.prompt,
        "story": story,
        "settings": {
            "length": req.length,
            "temperature": req.temperature,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty
        }
    }