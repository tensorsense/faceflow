from onnxruntime import InferenceSession
import numpy as np
from typing import Dict


def get_session(onnx_path: str, provider: str = "CUDAExecutionProvider") -> InferenceSession:
    return InferenceSession(onnx_path, providers=[provider])


def infer(session: InferenceSession, x: np.ndarray, out_map: Dict[str, int] = None) -> Dict[str, np.ndarray]:
    if out_map is None:
        out_map = {"ord": 0}
    # compute ONNX Runtime output prediction
    ort_inputs = {session.get_inputs()[0].name: x}
    ort_outs = session.run(None, ort_inputs)

    return {head: ort_outs[idx] for head, idx in out_map.items()}


def ord_to_reg(x: np.ndarray, thresh: float = 0.5):
    n_cl, logits_per_class = x.shape
    step = 1 / logits_per_class

    x = 1 / (1 + np.exp(-x))
    x = (x > thresh).astype(np.float32)
    x = np.sum(x, axis=1)
    x = x * step
    return x



