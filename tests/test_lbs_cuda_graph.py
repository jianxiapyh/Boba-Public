"""Automatic LBS dispatch must preserve cache decisions and output ownership."""
import unittest
from unittest.mock import patch

import torch


DYNAMIC = ("R_cache", "F_prev", "rotation_computed", "Q_cache_bm", "motions_bm_fp32")


def make_cache(batch=1):
    generator = torch.Generator(device="cuda").manual_seed(173)
    bones = torch.randn(12, 3, device="cuda", generator=generator) * 0.03
    points = torch.randn(28, 3, device="cuda", generator=generator) * 0.02
    relations = du.get_topk_indices(bones, K=3)
    weights, indices = du.knn_weights_sparse(bones, points, K=3)
    quats = torch.nn.functional.normalize(torch.randn(28, 4, device="cuda", generator=generator), dim=-1)
    return du.build_rotation_reuse_cache(
        indices, weights, relations, bones, points, quats, torch.device("cuda"),
        len(bones), len(points), batch,
    )


def duplicate(cache):
    return {key: (value.clone() if key in DYNAMIC else value)
            for key, value in cache.items() if not key.startswith("_lbs_")}


def frames(cache):
    batch = cache["number_of_instance"]
    rest = cache["mass_nodes_rest"].repeat(batch, 1)
    moved = rest.clone().view(batch, -1, 3)
    moved[:, 2:5, 0] += 0.012
    moved[::2, 7, 2] -= 0.007
    return [rest, rest.clone(), moved.flatten(0, 1), moved.flatten(0, 1).clone(), rest]


def assert_equal(actual, expected):
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def reference_lbs(nodes, cache, **kwargs):
    # Test-only reference evaluation; production has no execution-mode selector.
    with patch("gaussian_splatting.lbs_cuda_graph.get_lbs_graph", return_value=None):
        return du.lbs_with_rotation_reuse(nodes, cache, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LBSCudaGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global du
        from gaussian_splatting import dynamic_utils as du

    def test_changing_nodes_and_reuse_state_are_exact_across_cutoff(self):
        for batch in (1, 4, 64, 65, 128):
            with self.subTest(batch=batch):
                eager = make_cache(batch)
                candidate = duplicate(eager)
                for nodes in frames(eager):
                    expected = reference_lbs(nodes, eager)
                    actual = du.lbs_with_rotation_reuse(nodes, candidate, copy_outputs=False)
                    assert_equal(actual, expected)
                    assert_equal([candidate[key] for key in DYNAMIC], [eager[key] for key in DYNAMIC])
                if batch <= 64:
                    self.assertEqual(len(candidate["_lbs_cuda_graphs"]), 1)
                else:
                    self.assertNotIn("_lbs_cuda_graphs", candidate)

    def test_borrowed_storage_reuse_and_owned_output_lifetime(self):
        cache = make_cache(4)
        inputs = frames(cache)
        owned = du.lbs_with_rotation_reuse(inputs[0], cache)
        saved = tuple(t.clone() for t in owned)
        borrowed = du.lbs_with_rotation_reuse(inputs[0], cache, copy_outputs=False)
        pointers = tuple(t.data_ptr() for t in borrowed)
        changed = du.lbs_with_rotation_reuse(inputs[2], cache, copy_outputs=False)
        self.assertEqual(pointers, tuple(t.data_ptr() for t in changed))
        assert_equal(owned, saved)
        self.assertFalse(torch.equal(changed[0], saved[0]))

    def test_large_batch_outputs_survive_the_next_call(self):
        cache = make_cache(65)
        inputs = frames(cache)
        first = du.lbs_with_rotation_reuse(inputs[0], cache, copy_outputs=False)
        saved = tuple(t.clone() for t in first)
        changed = du.lbs_with_rotation_reuse(inputs[2], cache, copy_outputs=False)
        self.assertNotIn("_lbs_cuda_graphs", cache)
        assert_equal(first, saved)
        self.assertFalse(torch.equal(changed[0], saved[0]))

    def test_new_allocation_and_threshold_rebuild_graph(self):
        cache = make_cache()
        nodes = frames(cache)[0]
        du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
        previous = next(iter(cache["_lbs_cuda_graphs"].values()))
        cache["F_prev"] = cache["F_prev"].clone()
        expected_cache = duplicate(cache)
        actual = du.lbs_with_rotation_reuse(nodes, cache, tau_F=0.0, copy_outputs=False)
        current = next(iter(cache["_lbs_cuda_graphs"].values()))
        expected = reference_lbs(nodes, expected_cache, tau_F=0.0)
        self.assertIsNot(previous, current)
        self.assertEqual(len(cache["_lbs_cuda_graphs"]), 1)
        assert_equal(actual, expected)
        assert_equal([cache[key] for key in DYNAMIC], [expected_cache[key] for key in DYNAMIC])

    def test_precision_change_and_nondefault_stream(self):
        cache = make_cache()
        original = torch.get_float32_matmul_precision()
        try:
            nodes = frames(cache)[0]
            du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
            previous = next(iter(cache["_lbs_cuda_graphs"].values()))
            torch.set_float32_matmul_precision("highest" if original != "highest" else "high")
            du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
            self.assertIsNot(previous, next(iter(cache["_lbs_cuda_graphs"].values())))
            nodes = frames(cache)[2]
            reference_cache = duplicate(cache)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
            torch.cuda.current_stream().wait_stream(stream)
            expected = reference_lbs(nodes, reference_cache)
            assert_equal(actual, expected)
            self.assertEqual(len(cache["_lbs_cuda_graphs"]), 2)
        finally:
            torch.set_float32_matmul_precision(original)

    def test_inference_context_transition(self):
        cache = make_cache()
        nodes = frames(cache)[0]
        with torch.inference_mode():
            first = du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
        actual = du.lbs_with_rotation_reuse(nodes, cache, copy_outputs=False)
        expected = reference_lbs(nodes, duplicate(cache))
        assert_equal(actual, expected)
        assert_equal(first, actual)


if __name__ == "__main__":
    unittest.main()
