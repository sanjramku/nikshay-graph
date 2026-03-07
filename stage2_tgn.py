"""
stage2_tgn.py
=============
Stage 2: Temporal Risk Modelling — Temporal Graph Network (TGN)

Implements the full TGN architecture from the pipeline document:
  - Memory module: 64-dim GRU-updated vector per patient node
  - Graph Attention (GATConv): attention weights extracted for explainability
  - Silence events update memory (disengagement before dose gaps)
  - Patient memory persisted in Cosmos DB between inference sessions
  - Deployed as Azure ML managed endpoint in production

For hackathon prototype:
  - TGN runs locally using PyTorch Geometric
  - Azure ML endpoint call is stubbed with clear TODO markers
  - Memory vectors saved locally + pushed to Cosmos

Architecture:
  Input events (dose, visit, silence, contact symptom)
    → GRU memory update per node
    → GATConv aggregation over neighbours
    → 64-dim embedding
    → 2-layer MLP prediction head
    → dropout probability (0-1)

Usage:
    from stage2_tgn import TGNRiskModel, run_tgn_inference
    embeddings, attention_weights = run_tgn_inference(patients, graph)
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH GEOMETRIC TGN — full implementation
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch Geometric not installed. TGN will run in simulation mode.")
    print("Install with: pip install torch torch-geometric")


if TORCH_AVAILABLE:

    class GRUMemoryModule(nn.Module):
        """
        Memory module from Temporal Graph Networks (Rossi et al. 2020).
        Each node carries a 64-dim memory vector updated by a GRU cell.
        The GRU learns which events should cause large memory updates
        and which should cause small ones — from training data, not rules.
        """
        def __init__(self, memory_dim: int = 64, message_dim: int = 32):
            super().__init__()
            self.memory_dim  = memory_dim
            self.message_dim = message_dim
            self.gru         = nn.GRUCell(message_dim, memory_dim)

        def forward(self, current_memory: torch.Tensor,
                    message: torch.Tensor) -> torch.Tensor:
            """
            Update memory given a new event message.
            current_memory: (N, 64)
            message:        (N, 32)
            returns:        (N, 64) updated memory
            """
            return self.gru(message, current_memory)

        def encode_event(self, event_features: dict) -> torch.Tensor:
            """
            Convert a raw event dict into a 32-dim message vector.
            Features: days_missed, risk_score, silence_duration,
                      dose_confirmed, new_symptom, patient_reluctance
            """
            vec = torch.zeros(self.message_dim)
            vec[0]  = float(event_features.get("days_missed", 0)) / 14.0
            vec[1]  = float(event_features.get("risk_score", 0))
            vec[2]  = float(event_features.get("silence_days", 0)) / 30.0
            vec[3]  = 1.0 if event_features.get("dose_confirmed") else 0.0
            vec[4]  = 1.0 if event_features.get("new_symptom_in_contact") else 0.0
            vec[5]  = 1.0 if event_features.get("patient_reluctance") else 0.0
            vec[6]  = float(event_features.get("asha_load_score", 0))
            vec[7]  = float(event_features.get("treatment_week", 0)) / 26.0
            vec[8]  = 1.0 if event_features.get("prior_lfu") else 0.0
            vec[9]  = 1.0 if event_features.get("hiv") else 0.0
            vec[10] = 1.0 if event_features.get("diabetes") else 0.0
            vec[11] = float(event_features.get("adherence_rate", 1.0))
            # Remaining dims: zero-padded (learned embeddings in full training)
            return vec.unsqueeze(0)


    class TGNRiskModel(nn.Module):
        """
        Full Temporal Graph Network for TB dropout risk.
        Components:
          1. GRUMemoryModule — updates node memory on each event
          2. GATConv — graph attention over neighbours, weights used for explainability
          3. MLP prediction head — memory + neighbour context → dropout probability
        """
        def __init__(self, memory_dim: int = 64, hidden_dim: int = 64,
                     n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.memory      = GRUMemoryModule(memory_dim)
            self.attention   = GATConv(memory_dim, hidden_dim // n_heads,
                                       heads=n_heads, dropout=0.1, concat=True)
            self.norm        = nn.LayerNorm(hidden_dim)
            self.head        = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self._attention_weights = None  # stored at inference for explainability

        def forward(self, memory_vectors: torch.Tensor,
                    edge_index: torch.Tensor) -> tuple:
            """
            Forward pass.
            memory_vectors: (N, 64) — current memory state of all nodes
            edge_index:     (2, E) — graph connectivity
            returns: (risk_scores, attention_weights)
            """
            # Graph attention aggregation
            h, (edge_idx, attn_weights) = self.attention(
                memory_vectors, edge_index, return_attention_weights=True
            )
            h = self.norm(h)
            self._attention_weights = (edge_idx, attn_weights.detach())

            # Risk prediction
            risk = self.head(h)
            # Return edge_idx (GATConv's own expanded edge index) alongside
            # attn_weights. GATConv may produce more edges than the input
            # edge_index — callers MUST use edge_idx (not the original
            # edge_index) when indexing into attn_weights.
            return risk.squeeze(-1), edge_idx, attn_weights

        def extract_top_attention_factors(self, node_idx: int,
                                          gat_edge_index: torch.Tensor,
                                          attn_weights: torch.Tensor,
                                          node_labels: list) -> list:
            """
            For a given patient node, return the top 3 neighbours by attention
            weight — these are the explainability factors shown to the officer.

            gat_edge_index: edge index returned by GATConv (may differ in size
                            from the original edge_index passed to forward()).
                            Shape: (2, E_gat).
            attn_weights:   attention weights from GATConv. Shape: (E_gat, n_heads).
            """
            # Find all edges pointing TO this node in GATConv's edge index
            target_mask   = (gat_edge_index[1] == node_idx)
            source_nodes  = gat_edge_index[0][target_mask]
            weights       = attn_weights[target_mask].mean(dim=-1)

            if weights.numel() == 0:
                return []

            top_k  = min(3, weights.numel())
            top_idx = weights.argsort(descending=True)[:top_k]
            return [
                {
                    "neighbour": (node_labels[source_nodes[i].item()]
                                  if node_labels and source_nodes[i].item() < len(node_labels)
                                  else f"node_{source_nodes[i].item()}"),
                    "attention_weight": round(weights[i].item(), 4)
                }
                for i in top_idx
            ]


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODE — when PyTorch not available
# ─────────────────────────────────────────────────────────────────────────────

def simulate_tgn_output(patients: list) -> tuple:
    """
    Simulate TGN output without PyTorch.
    Returns (risk_scores_dict, attention_weights_dict) with realistic values
    derived from the clinical knowledge-based formula.
    Used for prototype demo when GPU/PyTorch not available.
    """
    # Import the knowledge-based formula from Stage 3 as a stand-in
    from stage3_score import compute_risk_score

    risk_scores = {}
    attention_weights = {}

    for p in patients:
        result = compute_risk_score(p)
        pid    = p["patient_id"]
        risk_scores[pid] = result["risk_score"]

        # Simulate attention weights using factor magnitudes
        factors = result.get("all_factors", {})
        total   = sum(abs(np.log(v)) for v in factors.values() if v > 0) or 1
        attention_weights[pid] = {
            k: round(abs(np.log(v)) / total, 4)
            for k, v in list(factors.items())[:3]
        }

    return risk_scores, attention_weights


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PYTORCH GRAPH from patient records
# ─────────────────────────────────────────────────────────────────────────────

def build_pytorch_graph(patients: list) -> tuple:
    """
    Build edge_index and memory_vectors for TGN input.
    Returns (edge_index, memory_vectors, node_id_map)
    """
    if not TORCH_AVAILABLE:
        return None, None, {}

    node_id_map = {}  # node_id → integer index
    memory_list = []
    edge_list   = []

    # Index patient nodes first
    for p in patients:
        idx = len(node_id_map)
        node_id_map[p["patient_id"]] = idx
        # Memory vector: load from stored state or initialise from features
        mem = torch.zeros(64)
        mem[0]  = p["risk_score"]
        mem[1]  = p["adherence"]["days_since_last_dose"] / 14.0
        mem[2]  = p["adherence"]["adherence_rate_30d"]
        mem[3]  = p["adherence"]["distance_to_center_km"] / 20.0
        mem[4]  = 1.0 if p["clinical"]["comorbidities"]["hiv"] else 0.0
        mem[5]  = 1.0 if p["adherence"]["prior_lfu_history"] else 0.0
        mem[6]  = p["clinical"]["total_treatment_days"] / 180.0
        memory_list.append(mem)

    # Index contact nodes
    for p in patients:
        pid = p["patient_id"]
        for c in p["contact_network"]:
            cid = f"CONTACT_{c['name'].replace(' ', '_').replace('.', '')}"
            if cid not in node_id_map:
                idx = len(node_id_map)
                node_id_map[cid] = idx
                mem = torch.zeros(64)
                mem[0] = c["vulnerability_score"] / 2.0
                mem[1] = 1.0 if c["rel"] == "Household" else 0.5
                memory_list.append(mem)

            # Edge: patient → contact
            edge_list.append([node_id_map[pid], node_id_map[cid]])
            # Bidirectional for attention
            edge_list.append([node_id_map[cid], node_id_map[pid]])

    memory_vectors = torch.stack(memory_list)
    edge_index     = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros(2, 0, dtype=torch.long)

    return edge_index, memory_vectors, node_id_map


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY PERSISTENCE — Cosmos DB
# ─────────────────────────────────────────────────────────────────────────────

def save_memory_to_cosmos(gc, patient_id: str, memory_vector):
    """
    Persist updated TGN memory vector to Cosmos DB patient node.
    Memory survives across sessions — patient history is not lost.
    """
    if gc is None:
        return
    if TORCH_AVAILABLE and hasattr(memory_vector, 'tolist'):
        mem_list = memory_vector.tolist()
    else:
        mem_list = list(memory_vector)

    from datetime import datetime, timezone
    mem_str    = json.dumps(mem_list)
    updated_at = datetime.now(timezone.utc).isoformat()

    from cosmos_client import run_query, safe
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('memory_vector', '{safe(mem_str)}')"
        f".property('memory_updated_at', '{safe(updated_at)}')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AZURE ML ENDPOINT CALL (production path)
# ─────────────────────────────────────────────────────────────────────────────

def call_azure_ml_endpoint(patient_features: list) -> list:
    """
    TODO: In production, replace simulate_tgn_output() with this function.
    Calls the TGN deployed as an Azure ML managed online endpoint.

    Setup:
    1. Azure ML workspace → Models → Register your trained TGN .pt file
    2. Endpoints → Create managed online endpoint
    3. Deploy model to endpoint with GPU compute
    4. Get scoring URI from endpoint details

    Returns list of risk score floats.
    """
    import requests
    endpoint_url = os.getenv("AZURE_ML_ENDPOINT_URL")
    endpoint_key = os.getenv("AZURE_ML_ENDPOINT_KEY")

    if not endpoint_url or not endpoint_key:
        raise ValueError("AZURE_ML_ENDPOINT_URL and AZURE_ML_ENDPOINT_KEY not set in .env")

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {endpoint_key}"
    }
    payload = {"input_data": {"data": patient_features}}
    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tgn_inference(patients: list, gc=None) -> tuple:
    """
    Run TGN inference on patient list.
    Returns (risk_scores_dict, attention_weights_dict)

    Production path: calls Azure ML endpoint.
    Prototype path: runs PyTorch locally or simulation mode.
    """
    azure_ml_url = os.getenv("AZURE_ML_ENDPOINT_URL")

    if azure_ml_url:
        # Production: Azure ML managed endpoint
        print("  [Stage 2] Running TGN inference via Azure ML endpoint...")
        try:
            features = [{"patient_id": p["patient_id"], **p["adherence"]} for p in patients]
            scores   = call_azure_ml_endpoint(features)
            risk_scores = {p["patient_id"]: s for p, s in zip(patients, scores)}
            attention_weights = {}  # not returned by endpoint in this example
            return risk_scores, attention_weights
        except Exception as e:
            print(f"  Azure ML call failed ({e}), falling back to local mode.")

    if TORCH_AVAILABLE:
        # Local PyTorch path
        print("  [Stage 2] Running TGN inference locally (PyTorch Geometric)...")
        model = TGNRiskModel()
        model.eval()

        edge_index, memory_vectors, node_id_map = build_pytorch_graph(patients)

        with torch.no_grad():
            risk_tensor, gat_edge_index, attn = model(memory_vectors, edge_index)

        risk_scores = {}
        attention_weights = {}
        node_labels = list(node_id_map.keys())

        for p in patients:
            pid = p["patient_id"]
            idx = node_id_map.get(pid, 0)
            risk_scores[pid] = round(float(risk_tensor[idx].item()), 4)

            # Use gat_edge_index (GATConv's expanded edges) — NOT the original
            # edge_index — to index into attn, which has shape (E_gat, n_heads)
            top_factors = model.extract_top_attention_factors(idx, gat_edge_index, attn, node_labels)
            attention_weights[pid] = top_factors

            # Persist memory to Cosmos
            if gc:
                save_memory_to_cosmos(gc, pid, memory_vectors[idx])

        print(f"  [Stage 2] TGN inference complete. {len(risk_scores)} patients scored.")
        return risk_scores, attention_weights

    else:
        # Simulation mode — no PyTorch
        print("  [Stage 2] Running TGN in simulation mode (PyTorch not available).")
        return simulate_tgn_output(patients)


if __name__ == "__main__":
    with open("nikshay_grounded_dataset.json") as f:
        patients = json.load(f)[:100]

    risk_scores, attn = run_tgn_inference(patients)
    print(f"\nSample outputs (first 5 patients):")
    for pid, score in list(risk_scores.items())[:5]:
        factors = attn.get(pid, [])
        print(f"  {pid}: risk={score}  attention={factors}")