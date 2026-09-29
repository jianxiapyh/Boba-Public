"""Replay the fixed-shape LBS stages around its dynamic rotation update."""

from __future__ import annotations

import torch


_CACHE_TENSORS = (
    "mass_nodes_rest", "gaussians_quat_rest", "relations", "weights_indices",
    "rest_bone_to_neighbors", "R_cache", "F_prev", "rotation_computed",
    "xyz_local_w", "bones_rest_blend", "W_csr_f32", "W_csr_f16",
    "Q_cache_bm", "motions_bm_fp32",
)
_CACHE_SCALARS = (
    "mass_nodes_per_instance", "gaussians_per_instance", "number_of_instance",
)


def _tensor_signature(tensor):
    metadata = (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.layout)
    if tensor.layout == torch.sparse_csr:
        return metadata + (
            tensor.crow_indices().data_ptr(), tensor.col_indices().data_ptr(),
            tensor.values().data_ptr(), tensor.values().numel(),
        )
    return metadata + (tensor.data_ptr(), tensor.stride())


class _LBSGraph:
    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, nodes, cache, tau_F, prepare, deform, signature):
        self.signature = signature
        # Hold allocations, not the owning dictionary containing this graph.
        self.bindings = {
            name: cache[name] for name in _CACHE_TENSORS + _CACHE_SCALARS
        }
        self.nodes = nodes.clone().contiguous()
        caller = torch.cuda.current_stream(nodes.device)
        stream = torch.cuda.Stream(device=nodes.device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            # Initialize library work/compiled quaternion kernels before capture.
            for _ in range(3):
                prepare(self.nodes, self.bindings, tau_F)
            self.prepare_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prepare_graph, stream=stream):
                self.prepared = prepare(self.nodes, self.bindings, tau_F)
            self.prepare_graph.replay()
            for _ in range(3):
                deform(self.prepared[0], self.bindings)
            self.deform_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.deform_graph, stream=stream):
                self.output = deform(self.prepared[0], self.bindings)
        caller.wait_stream(stream)

    def prepare(self, nodes):
        self.nodes.copy_(nodes)
        self.prepare_graph.replay()
        return self.prepared

    def deform(self, *, copy_outputs):
        self.deform_graph.replay()
        if copy_outputs:
            return self.output[0].clone(), self.output[1].clone()
        return self.output


def get_lbs_graph(nodes, cache, tau_F, prepare, deform):
    """Get graphs bound to this cache and stream, or keep eager execution.

    Rotation selection, its variable-size eigensolve and selective cache updates
    stay outside the graphs. They retain the reference algorithm and thresholds.
    Batches above 64 use ordinary CUDA kernel launches. Each stream keeps
    only its latest allocation/precision configuration.
    """
    if (
        cache["number_of_instance"] > 64
        or not nodes.is_cuda
        or nodes.dtype != torch.float32
        or torch.is_autocast_enabled("cuda")
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    with torch.cuda.device(nodes.device):
        stream = torch.cuda.current_stream(nodes.device).cuda_stream
        signature = (
            tuple(nodes.shape), nodes.dtype, nodes.device, float(tau_F),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            tuple(cache[name] for name in _CACHE_SCALARS),
            tuple(_tensor_signature(cache[name]) for name in _CACHE_TENSORS),
        )
        graphs = cache.setdefault("_lbs_cuda_graphs", {})
        entry = graphs.get(stream)
        if entry is None or entry.signature != signature:
            entry = _LBSGraph(nodes, cache, tau_F, prepare, deform, signature)
            graphs[stream] = entry
        return entry
