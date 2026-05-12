"""
Qwen3 Model Server
==================
Loads Qwen3 once, listens on port 8000, serves generation & interpretability endpoints.

Endpoints:
  POST /load                — load model from directory
  GET  /health              — check server status
  POST /generate            — text generation (supports SSE streaming)
  POST /hidden_states       — extract hidden states at specified layers
  POST /logits_lens         — project intermediate hidden states through LM head
  POST /sae                 — apply Sparse Autoencoder to extract features
  POST /sae_set_dir         — set SAE model directory
  GET  /sae_loaded_layers   — list layers with SAE loaded in cache

Usage:
  python qwen3_server.py
  # then POST /load with {"model_dir": "/path/to/Qwen3-8B"}
"""

import json
import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from threading import Thread

app = FastAPI(title="Qwen3 Server", version="0.1.0")

# --- global model state ---
model = None
tokenizer = None
lm_head = None
model_config = {}
model_loaded = False

# --- SAE state ---
sae_dir = None
sae_cache: dict[int, dict[str, torch.Tensor]] = {}


def load_model_impl(model_dir: str):
    global model, tokenizer, lm_head, model_config, model_loaded

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"Loading model (device={device}, dtype={dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Get LM head for logits lens projection
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = model.lm_head

    cfg = model.config
    model_config = {
        "device": device,
        "dtype": str(dtype),
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "model_type": cfg.model_type,
    }
    model_loaded = True
    print(f"Model loaded: {json.dumps(model_config, indent=2)}")
    return model_config


# --- Pydantic schemas ---

class LoadRequest(BaseModel):
    model_dir: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    stream: bool = False


class HiddenStatesRequest(BaseModel):
    prompt: str
    layers: list[int]
    """0-indexed layer indices to return. Pass [0, 5, 10, 20, 31] for Qwen3-8B."""
    token_position: int = -1
    """Token position to extract. -1 = last token."""


class LogitsLensRequest(BaseModel):
    prompt: str
    layers: list[int]
    top_k: int = 10
    token_position: int = -1


# --- Streaming helper ---

async def _stream_generate(gen_kwargs):
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs={**gen_kwargs, "streamer": streamer})
    thread.start()

    for text in streamer:
        if text:
            yield f"data: {json.dumps({'token': text})}\n\n"
    yield "data: [DONE]\n\n"


# ================================================================
#  Routes
# ================================================================

@app.get("/health")
def health():
    if not model_loaded:
        return {"status": "no_model_loaded"}
    return {"status": "ok", "config": model_config}


@app.post("/load")
def load(req: LoadRequest):
    cfg = load_model_impl(req.model_dir)
    return {"status": "ok", "config": cfg}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not model_loaded:
        raise HTTPException(503, "Model not loaded. POST /load first.")

    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature if req.temperature > 0 else None,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    if req.stream:
        return StreamingResponse(
            _stream_generate(gen_kwargs),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    with torch.no_grad():
        gen_ids = model.generate(**gen_kwargs)
    response = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"response": response}


@app.post("/hidden_states")
def get_hidden_states(req: HiddenStatesRequest):
    """
    Extract residual stream hidden states at specified layers.

    Returns the hidden state vector at the requested token position for each
    requested layer. Useful as input to a Sparse Autoencoder (SAE) or for
    activation analysis.
    """
    if not model_loaded:
        raise HTTPException(503, "Model not loaded.")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    n_layers = model.config.num_hidden_layers
    # hidden_states[0] = embeddings, hidden_states[L+1] = layer L output
    seq_len = inputs.input_ids.shape[1]
    pos = min(seq_len - 1, req.token_position) if req.token_position >= 0 else seq_len - 1

    result = {}
    for layer_idx in req.layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise HTTPException(400, f"Layer {layer_idx} out of range [0, {n_layers-1}]")
        hs = outputs.hidden_states[layer_idx + 1][0, pos, :].cpu().to(torch.float32)
        result[str(layer_idx)] = hs.tolist()

    return {
        "prompt": req.prompt,
        "num_tokens": seq_len,
        "token_position": pos,
        "hidden_size": model.config.hidden_size,
        "layers": result,
    }


@app.post("/logits_lens")
def logits_lens(req: LogitsLensRequest):
    """
    Logits Lens: project each intermediate layer's hidden state through the
    LM head to see what the model "predicts" at that layer.

    Returns top-k tokens and their probabilities for each requested layer.
    """
    if not model_loaded:
        raise HTTPException(503, "Model not loaded.")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    n_layers = model.config.num_hidden_layers
    seq_len = inputs.input_ids.shape[1]
    pos = min(seq_len - 1, req.token_position) if req.token_position >= 0 else seq_len - 1

    result = {}
    for layer_idx in req.layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise HTTPException(400, f"Layer {layer_idx} out of range [0, {n_layers-1}]")

        hs = outputs.hidden_states[layer_idx + 1][:, pos:pos+1, :]  # (1, 1, hidden)
        logits = lm_head(hs)  # (1, 1, vocab_size)
        probs = torch.softmax(logits[0, 0], dim=-1)

        top_probs, top_indices = torch.topk(probs, req.top_k)
        top_tokens = tokenizer.batch_decode(top_indices.unsqueeze(-1))

        result[str(layer_idx)] = [
            {"token": tok, "token_id": idx.item(), "prob": round(p.item(), 6)}
            for tok, idx, p in zip(top_tokens, top_indices, top_probs)
        ]

    return {
        "prompt": req.prompt,
        "num_tokens": seq_len,
        "token_position": pos,
        "layers": result,
    }


# ================================================================
#  SAE
# ================================================================

class SAESetDirRequest(BaseModel):
    sae_dir: str


class SAERequest(BaseModel):
    prompt: str
    layers: list[int]
    """Layer indices to run SAE on. Each will be loaded on-demand."""
    token_position: int = -1
    """Token position to analyze. -1 = last token."""
    max_features: int = 20
    """Number of top activating features to return per layer."""
    include_reconstruction: bool = False
    """Also return reconstructed residual and MSE loss."""


class InterventionSpec(BaseModel):
    layer: int
    feature_id: int
    action: str = "zero"
    """One of: "zero", "scale", "set", "clamp_max"."""
    value: float = 0.0
    """Scale factor (for "scale") or target activation (for "set"/"clamp_max")."""


class SAEInterventionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    interventions: list[InterventionSpec]
    stream: bool = False


def _load_sae(layer: int):
    """Load a single SAE checkpoint into cache (if not already loaded)."""
    global sae_dir, sae_cache
    if sae_dir is None:
        raise HTTPException(400, "SAE dir not set. POST /sae_set_dir first.")
    if layer in sae_cache:
        return sae_cache[layer]
    path = os.path.join(sae_dir, f"layer{layer}.sae.pt")
    if not os.path.exists(path):
        raise HTTPException(400, f"SAE file not found for layer {layer}: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    dev = next(model.parameters()).device
    dtype_ = next(model.parameters()).dtype
    for k in state:
        state[k] = state[k].to(device=dev, dtype=dtype_)
    sae_cache[layer] = state
    return state


def _run_sae(residual: torch.Tensor, sae_state: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    residual: (d_model,) tensor on model device
    Returns (active_indices, active_values, reconstructed)
    """
    W_enc = sae_state["W_enc"]
    b_enc = sae_state["b_enc"]
    W_dec = sae_state["W_dec"]
    b_dec = sae_state["b_dec"]

    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(pre_acts, k=100, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    reconstructed = acts @ W_dec.T + b_dec

    return topk_idx, topk_vals, reconstructed


def _intervene_sae(
    residual: torch.Tensor,
    sae_state: dict,
    specs: list[dict],
) -> tuple[torch.Tensor, dict]:
    """
    Apply SAE interventions on one residual vector.

    Supports two families of actions:
      - Old-style (zero/negate/scale/set/clamp_max): modify SAE acts, then decode.
      - add_direction: encode to check activation, then directly add/subtract
        the W_dec direction vector in residual space (no decode loss).

    Returns (modified_residual, {fid: ..., ...}).
    """
    W_enc = sae_state["W_enc"]
    b_enc = sae_state["b_enc"]
    W_dec = sae_state["W_dec"]
    b_dec = sae_state["b_dec"]

    # Encode — always needed to check which features are active
    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(pre_acts, k=100, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)

    # Build a set of active feature IDs for fast membership check
    active_set = set(topk_idx.tolist())

    log = {}
    direction_delta = None  # accumulator for add_direction modifications

    for spec in specs:
        fid = spec["feature_id"]
        orig = acts[fid].item()
        val = spec["value"]

        if spec["action"] == "add_direction":
            # Only intervene if this feature is actually active
            if fid in active_set:
                if direction_delta is None:
                    direction_delta = torch.zeros_like(residual)
                acts[fid] = 0.0                      # zero SAE contribution to avoid double-count
                direction_delta += val * W_dec[:, fid]  # α * direction in residual space (W_dec: 4096×65536)
                log[str(fid)] = {"original": round(orig, 4), "new": 0.0,
                                 "action": "add_direction", "alpha": val}
            else:
                log[str(fid)] = {"original": 0.0, "new": 0.0,
                                 "action": "add_direction_skipped", "reason": "not_in_topk"}
        elif spec["action"] == "zero":
            acts[fid] = 0.0
            log[str(fid)] = {"original": round(orig, 4), "new": 0.0,
                             "action": "zero", "value": 0.0}
        elif spec["action"] == "negate":
            acts[fid] = -acts[fid]
            log[str(fid)] = {"original": round(orig, 4), "new": round(acts[fid].item(), 4),
                             "action": "negate", "value": 0.0}
        elif spec["action"] == "scale":
            acts[fid] = acts[fid] * val
            log[str(fid)] = {"original": round(orig, 4), "new": round(acts[fid].item(), 4),
                             "action": "scale", "value": val}
        elif spec["action"] == "set":
            acts[fid] = val
            log[str(fid)] = {"original": round(orig, 4), "new": val,
                             "action": "set", "value": val}
        elif spec["action"] == "clamp_max":
            acts[fid] = min(acts[fid].item(), val)
            log[str(fid)] = {"original": round(orig, 4), "new": round(acts[fid].item(), 4),
                             "action": "clamp_max", "value": val}
        else:
            raise HTTPException(400, f"Unknown action: {spec['action']}")

    # Decode — may be skip-able if only add_direction actions were applied
    # and none of those features were active (direction_delta is None).
    # But the safe path is to always decode to handle mixed specs correctly.
    reconstructed = acts @ W_dec.T + b_dec
    if direction_delta is not None:
        result = reconstructed + direction_delta
    else:
        result = reconstructed

    return result, log


def _make_intervention_hook(interventions: dict[int, list[dict]], sae_states: dict[int, dict]):
    """
    Returns a forward hook that applies SAE feature intervention at all positions.
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        clone = hidden.clone()
        logs = {}
        for pos in range(clone.shape[1]):
            new_resid, pos_log = _intervene_sae(clone[0, pos], sae_states, interventions)
            clone[0, pos] = new_resid
            logs[f"pos_{pos}"] = pos_log
        if isinstance(output, tuple):
            return (clone,) + output[1:]
        return clone
    return hook


@app.post("/sae_set_dir")
def sae_set_dir(req: SAESetDirRequest):
    global sae_dir
    abs_path = os.path.abspath(req.sae_dir)
    if not os.path.isdir(abs_path):
        raise HTTPException(400, f"Directory not found: {abs_path}")
    sae_dir = abs_path
    sae_cache.clear()
    available = []
    for fname in sorted(os.listdir(abs_path)):
        if fname.startswith("layer") and fname.endswith(".sae.pt"):
            try:
                layer = int(fname[len("layer"):-len(".sae.pt")])
                available.append(layer)
            except ValueError:
                pass
    available.sort()
    return {"status": "ok", "sae_dir": abs_path, "available_layers": available, "num_layers": len(available)}


@app.get("/sae_loaded_layers")
def sae_loaded_layers():
    return {"loaded_layers": sorted(sae_cache.keys())}


@app.post("/sae")
def sae_inference(req: SAERequest):
    """Run SAE on hidden states at specified layers."""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded.")
    if sae_dir is None:
        raise HTTPException(400, "SAE dir not set. POST /sae_set_dir first.")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    n_layers = model.config.num_hidden_layers
    seq_len = inputs.input_ids.shape[1]
    pos = min(seq_len - 1, req.token_position) if req.token_position >= 0 else seq_len - 1

    result = {}
    for layer_idx in req.layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise HTTPException(400, f"Layer {layer_idx} out of range [0, {n_layers-1}]")
        sae_state = _load_sae(layer_idx)
        residual = outputs.hidden_states[layer_idx + 1][0, pos, :]
        topk_idx, topk_vals, reconstructed = _run_sae(residual, sae_state)

        top_n = min(req.max_features, 100)
        sorted_vals, sorted_idx = topk_vals.sort(descending=True)
        features = [{"feature_id": int(topk_idx[sorted_idx[i].item()].item()), "activation": round(sorted_vals[i].item(), 4)} for i in range(top_n)]

        entry = {"prompt": req.prompt, "num_tokens": seq_len, "token_position": pos, "layer": layer_idx, "feature_count": 100, "top_features": features, "residual_norm": round(residual.norm().item(), 4)}
        if req.include_reconstruction:
            loss = (reconstructed - residual).pow(2).mean().item()
            entry["reconstruction"] = {"mse_loss": round(loss, 6), "reconstructed_norm": round(reconstructed.norm().item(), 4)}
        result[str(layer_idx)] = entry

    return {"layers": result}


@app.post("/sae_intervene")
async def sae_intervene(req: SAEInterventionRequest):
    """
    Generate text with SAE feature interventions.

    Registers forward hooks on the specified layers that decode the residual
    into SAE features, apply user interventions (zero/scale/set/clamp_max),
    then re-encode the modified residual back — all during the generation
    forward pass. Hooks are removed when generation finishes.
    """
    if not model_loaded:
        raise HTTPException(503, "Model not loaded.")
    if sae_dir is None:
        raise HTTPException(400, "SAE dir not set. POST /sae_set_dir first.")

    from collections import defaultdict
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for spec in req.interventions:
        by_layer[spec.layer].append({"feature_id": spec.feature_id, "action": spec.action, "value": spec.value})

    # Register hooks
    hooks = []
    try:
        for layer_idx, specs in by_layer.items():
            if layer_idx < 0 or layer_idx >= model.config.num_hidden_layers:
                raise HTTPException(400, f"Layer {layer_idx} out of range [0, {model.config.num_hidden_layers-1}]")
            sae_state = _load_sae(layer_idx)
            hook_fn = _make_intervention_hook(specs, sae_state)
            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(handle)

        messages = [{"role": "user", "content": req.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        gen_kwargs = dict(**inputs, max_new_tokens=req.max_new_tokens, do_sample=req.temperature > 0,
                          temperature=req.temperature if req.temperature > 0 else None,
                          top_p=req.top_p, top_k=req.top_k)

        if req.stream:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = Thread(target=model.generate, kwargs={**gen_kwargs, "streamer": streamer})
            thread.start()

            async def _gen():
                for tok in streamer:
                    if tok:
                        yield f"data: {json.dumps({'token': tok})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

        with torch.no_grad():
            gen_ids = model.generate(**gen_kwargs)
        response = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"response": response}

    finally:
        for h in hooks:
            h.remove()


# ================================================================
#  Entry point
# ================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
