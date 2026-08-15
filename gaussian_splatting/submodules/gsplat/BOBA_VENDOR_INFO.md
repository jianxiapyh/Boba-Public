# Boba Vendored `gsplat`

- Upstream repository: `https://github.com/nerfstudio-project/gsplat`
- Upstream tag: `v1.5.3`
- Upstream commit: `937e29912570c372bed6747a5c9bf85fed877bae`
- Vendored for Boba under: `gaussian_splatting/submodules/gsplat`
- Local patch marker: apply all Boba-specific changes after this snapshot

This vendored copy is pinned to the same `gsplat` version currently known to
work inside the `phystwin` conda environment. Boba installs it with:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 BUILD_NO_CUDA=1 \
  python -m pip install -e ./gaussian_splatting/submodules/gsplat
```

The editable install is scoped to the `phystwin` environment and Boba validates
that runtime imports resolve back to this vendored source tree.
