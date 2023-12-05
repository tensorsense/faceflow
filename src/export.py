import torch
import onnx
import onnxruntime
import numpy as np

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

from params.model import model, ckpt_path
import params.export as cfg


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict(x)


def trace(torch_model, onnx_path="model.onnx", batch_size=1, input_size=128):
    torch_model.train()

    # Input to the model
    x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def check_schema(onnx_path="model.onnx"):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def check_runtime(torch_model, onnx_path="model.onnx", batch_size=1, input_size=128):
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)
    torch_out = torch_model(x)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out["ord"]), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main() -> None:
    export_model = ExportModel(model.load_from_checkpoint(ckpt_path))

    trace(model, cfg.onnx_path, cfg.batch_size, cfg.input_size)
    check_schema(cfg.onnx_path)
    check_runtime(export_model, cfg.onnx_path, cfg.batch_size, cfg.input_size)


if __name__ == "__main__":
    main()



