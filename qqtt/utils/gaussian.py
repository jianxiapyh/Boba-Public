import copy

import torch


def remove_gaussians_with_low_opacity(gaussians, opacity_threshold=0.1):
    opacity = gaussians.get_opacity.squeeze(-1)
    mask3d = opacity > opacity_threshold
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with opacity < 0.1")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def _validate_instance_id(number_of_instance, instance_id):
    if instance_id is None:
        raise ValueError("instance_id is required for instance-selective rendering.")

    instance_id = int(instance_id)
    if instance_id < 0 or instance_id >= int(number_of_instance):
        raise ValueError(
            f"instance_id must be in [0, {int(number_of_instance) - 1}], got {instance_id}"
        )
    return instance_id


GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE = "shared_template"
GAUSSIAN_RENDER_MODE_DUPLICATED = "duplicated"
GAUSSIAN_RENDER_MODES = (
    GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE,
    GAUSSIAN_RENDER_MODE_DUPLICATED,
)


def normalize_gaussian_render_mode(gaussian_render_mode):
    if gaussian_render_mode is None:
        return GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE
    if gaussian_render_mode not in GAUSSIAN_RENDER_MODES:
        raise ValueError(
            "gaussian_render_mode must be one of "
            f"{GAUSSIAN_RENDER_MODES}. Received: {gaussian_render_mode}"
        )
    return gaussian_render_mode


class SharedGaussianBatchState:
    uses_shared_template_rendering = True

    def __init__(self, template_gaussians, instance_offsets):
        if instance_offsets.ndim != 2 or instance_offsets.shape[1] != 3:
            raise ValueError(
                f"instance_offsets must have shape (N, 3), got {tuple(instance_offsets.shape)}"
            )

        self.template_gaussians = template_gaussians
        self.number_of_instance = int(instance_offsets.shape[0])
        self.gaussians_per_instance = int(template_gaussians._xyz.shape[0])
        self.max_sh_degree = template_gaussians.max_sh_degree
        self.active_sh_degree = template_gaussians.active_sh_degree
        self.isotropic = template_gaussians.isotropic
        self.rotation_activation = template_gaussians.rotation_activation

        offsets = instance_offsets.to(
            device=template_gaussians._xyz.device,
            dtype=template_gaussians._xyz.dtype,
        ).contiguous()
        self.instance_offsets = offsets
        xyz_single = template_gaussians._xyz
        rotation_single = template_gaussians._rotation

        self._xyz = (
            xyz_single.unsqueeze(0) + offsets[:, None, :]
        ).reshape(-1, 3).contiguous()
        self._rotation = (
            rotation_single.unsqueeze(0)
            .repeat(self.number_of_instance, 1, 1)
            .reshape(-1, rotation_single.shape[-1])
            .contiguous()
        )

        expected_total = self.gaussians_per_instance * self.number_of_instance
        assert self._xyz.shape[0] == expected_total
        assert self._rotation.shape[0] == expected_total

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_template_features(self):
        return self.template_gaussians.get_features

    @property
    def get_template_opacity(self):
        return self.template_gaussians.get_opacity

    @property
    def get_template_scaling(self):
        return self.template_gaussians.get_scaling

    @property
    def xyz_rest_single(self):
        return self.template_gaussians._xyz

    @property
    def rotation_rest_single(self):
        return self.template_gaussians.get_rotation

    def get_instance_slice(self, instance_id):
        instance_id = _validate_instance_id(self.number_of_instance, instance_id)
        start = instance_id * self.gaussians_per_instance
        end = start + self.gaussians_per_instance
        return slice(start, end)

    def get_instance_offset(self, instance_id):
        instance_id = _validate_instance_id(self.number_of_instance, instance_id)
        return self.instance_offsets[instance_id]


class RenderCompatibleGaussianView:
    def __init__(self, shared_state: SharedGaussianBatchState):
        self.shared_state = shared_state
        template = shared_state.template_gaussians

        self.max_sh_degree = shared_state.max_sh_degree
        self.active_sh_degree = shared_state.active_sh_degree
        self.isotropic = shared_state.isotropic
        self.gaussians_per_instance = shared_state.gaussians_per_instance
        self.number_of_instance = shared_state.number_of_instance

        self.scaling_activation = template.scaling_activation
        self.covariance_activation = template.covariance_activation
        self.opacity_activation = template.opacity_activation
        self.rotation_activation = template.rotation_activation

        repeat_count = shared_state.number_of_instance
        self._features_dc = template._features_dc.repeat(
            repeat_count, 1, 1
        ).contiguous()
        self._features_rest = template._features_rest.repeat(
            repeat_count, 1, 1
        ).contiguous()
        self._opacity = template._opacity.repeat(repeat_count, 1).contiguous()
        self._scaling = template._scaling.repeat(repeat_count, 1).contiguous()

        expected_total = (
            shared_state.gaussians_per_instance * shared_state.number_of_instance
        )
        assert self._features_dc.shape[0] == expected_total
        assert self._features_rest.shape[0] == expected_total
        assert self._opacity.shape[0] == expected_total
        assert self._scaling.shape[0] == expected_total

    @property
    def _xyz(self):
        return self.shared_state._xyz

    @_xyz.setter
    def _xyz(self, value):
        self.shared_state._xyz = value

    @property
    def _rotation(self):
        return self.shared_state._rotation

    @_rotation.setter
    def _rotation(self, value):
        self.shared_state._rotation = value

    @property
    def get_xyz(self):
        return self.shared_state._xyz

    @property
    def get_rotation(self):
        return self.rotation_activation(self.shared_state._rotation)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        if self.isotropic:
            return scaling.repeat(1, 3)
        return scaling

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.shared_state._rotation
        )


class InstanceSelectiveGaussianView:
    def __init__(self, shared_state: SharedGaussianBatchState, instance_id):
        self.shared_state = shared_state
        template = shared_state.template_gaussians
        self.instance_id = _validate_instance_id(
            shared_state.number_of_instance, instance_id
        )
        self.instance_slice = shared_state.get_instance_slice(self.instance_id)
        self.instance_offset = shared_state.get_instance_offset(self.instance_id)

        self.max_sh_degree = shared_state.max_sh_degree
        self.active_sh_degree = shared_state.active_sh_degree
        self.isotropic = shared_state.isotropic
        self.gaussians_per_instance = shared_state.gaussians_per_instance
        self.number_of_instance = 1

        self.scaling_activation = template.scaling_activation
        self.covariance_activation = template.covariance_activation
        self.opacity_activation = template.opacity_activation
        self.rotation_activation = template.rotation_activation

        self._features_dc = template._features_dc
        self._features_rest = template._features_rest
        self._opacity = template._opacity
        self._scaling = template._scaling

    @property
    def _xyz(self):
        return self.shared_state._xyz[self.instance_slice]

    @property
    def _rotation(self):
        return self.shared_state._rotation[self.instance_slice]

    @property
    def get_xyz(self):
        return self.shared_state._xyz[self.instance_slice] - self.instance_offset

    @property
    def get_rotation(self):
        return self.rotation_activation(self.shared_state._rotation[self.instance_slice])

    @property
    def get_template_features(self):
        return self.shared_state.get_template_features

    @property
    def get_template_opacity(self):
        return self.shared_state.get_template_opacity

    @property
    def get_template_scaling(self):
        return self.shared_state.get_template_scaling

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        if self.isotropic:
            return scaling.repeat(1, 3)
        return scaling

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.shared_state._rotation[self.instance_slice]
        )


class SharedTemplateInstanceSelectiveGaussianView(InstanceSelectiveGaussianView):
    uses_shared_template_rendering = True


class SharedTemplateBatchImagesGaussianView:
    uses_shared_template_rendering = True
    uses_batch_image_rendering = True

    def __init__(self, shared_state: SharedGaussianBatchState):
        self.shared_state = shared_state
        template = shared_state.template_gaussians

        self.max_sh_degree = shared_state.max_sh_degree
        self.active_sh_degree = shared_state.active_sh_degree
        self.isotropic = shared_state.isotropic
        self.gaussians_per_instance = shared_state.gaussians_per_instance
        self.number_of_instance = shared_state.number_of_instance

        self.scaling_activation = template.scaling_activation
        self.covariance_activation = template.covariance_activation
        self.opacity_activation = template.opacity_activation
        self.rotation_activation = template.rotation_activation

        self._offsets_per_gaussian = (
            shared_state.instance_offsets[:, None, :]
            .expand(-1, self.gaussians_per_instance, -1)
            .reshape(-1, 3)
            .contiguous()
        )

    @property
    def get_xyz(self):
        return self.shared_state._xyz - self._offsets_per_gaussian

    @property
    def get_rotation(self):
        return self.rotation_activation(self.shared_state._rotation)

    @property
    def get_template_features(self):
        return self.shared_state.get_template_features

    @property
    def get_template_opacity(self):
        return self.shared_state.get_template_opacity

    @property
    def get_template_scaling(self):
        return self.shared_state.get_template_scaling

    @property
    def get_features(self):
        return self.shared_state.get_template_features

    @property
    def get_features_dc(self):
        return self.shared_state.template_gaussians._features_dc

    @property
    def get_features_rest(self):
        return self.shared_state.template_gaussians._features_rest

    @property
    def get_opacity(self):
        return self.shared_state.get_template_opacity

    @property
    def get_scaling(self):
        return self.shared_state.get_template_scaling

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.shared_state._rotation
        )


class DuplicatedBatchImagesGaussianView:
    uses_batch_image_rendering = True

    def __init__(self, shared_state: SharedGaussianBatchState):
        self.shared_state = shared_state
        template = shared_state.template_gaussians

        self.max_sh_degree = shared_state.max_sh_degree
        self.active_sh_degree = shared_state.active_sh_degree
        self.isotropic = shared_state.isotropic
        self.gaussians_per_instance = shared_state.gaussians_per_instance
        self.number_of_instance = shared_state.number_of_instance

        self.scaling_activation = template.scaling_activation
        self.covariance_activation = template.covariance_activation
        self.opacity_activation = template.opacity_activation
        self.rotation_activation = template.rotation_activation

        self._features_dc = template._features_dc.unsqueeze(0).repeat(
            self.number_of_instance, 1, 1, 1
        ).contiguous()
        self._features_rest = template._features_rest.unsqueeze(0).repeat(
            self.number_of_instance, 1, 1, 1
        ).contiguous()
        self._opacity = template._opacity.unsqueeze(0).repeat(
            self.number_of_instance, 1, 1
        ).contiguous()
        self._scaling = template._scaling.unsqueeze(0).repeat(
            self.number_of_instance, 1, 1
        ).contiguous()
        self._offsets_per_gaussian = shared_state.instance_offsets[:, None, :]

    @property
    def get_xyz(self):
        xyz = self.shared_state._xyz.reshape(
            self.number_of_instance, self.gaussians_per_instance, 3
        )
        return xyz - self._offsets_per_gaussian

    @property
    def get_rotation(self):
        rotations = self.shared_state._rotation.reshape(
            self.number_of_instance, self.gaussians_per_instance, -1
        )
        return self.rotation_activation(rotations)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        if self.isotropic:
            return scaling.repeat(1, 1, 3)
        return scaling

    def get_covariance(self, scaling_modifier=1):
        rotations = self.shared_state._rotation.reshape(
            self.number_of_instance, self.gaussians_per_instance, -1
        )
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, rotations
        )


def build_instance_selective_render_view(
    shared_state,
    instance_id,
    gaussian_render_mode=GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE,
):
    gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
    if gaussian_render_mode == GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE:
        return SharedTemplateInstanceSelectiveGaussianView(shared_state, instance_id)
    return InstanceSelectiveGaussianView(shared_state, instance_id)


def build_batch_images_render_view(
    shared_state,
    gaussian_render_mode=GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE,
):
    gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
    if gaussian_render_mode == GAUSSIAN_RENDER_MODE_DUPLICATED:
        return DuplicatedBatchImagesGaussianView(shared_state)
    return SharedTemplateBatchImagesGaussianView(shared_state)


def _build_instance_offsets(number_of_instance, offset_step, device, dtype):
    if number_of_instance < 1:
        raise ValueError(
            f"number_of_instance must be positive, got {number_of_instance}"
        )

    offset_step = offset_step.to(device=device, dtype=dtype).reshape(1, 3)
    instance_ids = torch.arange(number_of_instance, device=device, dtype=dtype).unsqueeze(1)
    return (instance_ids * offset_step).contiguous()


def _resolve_instance_offsets(
    number_of_instance,
    template_gaussians,
    offset_step=None,
    instance_offsets=None,
):
    if instance_offsets is not None:
        offsets = instance_offsets.to(
            device=template_gaussians._xyz.device,
            dtype=template_gaussians._xyz.dtype,
        ).contiguous()
        expected_shape = (int(number_of_instance), 3)
        if tuple(offsets.shape) != expected_shape:
            raise ValueError(
                f"instance_offsets must have shape {expected_shape}, got {tuple(offsets.shape)}"
            )
        return offsets

    if offset_step is None:
        raise ValueError("Either offset_step or instance_offsets must be provided.")

    return _build_instance_offsets(
        number_of_instance=number_of_instance,
        offset_step=offset_step,
        device=template_gaussians._xyz.device,
        dtype=template_gaussians._xyz.dtype,
    )


def load_gaussian_template(gs_path, sh_degree=3, opacity_threshold=0.1):
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(gs_path)
    gaussians = remove_gaussians_with_low_opacity(
        gaussians, opacity_threshold=opacity_threshold
    )
    gaussians.isotropic = True
    return gaussians


def load_shared_batched_gaussians(
    gs_path,
    number_of_instance,
    offset_step=None,
    instance_offsets=None,
    sh_degree=3,
    opacity_threshold=0.1,
    gaussian_render_mode=GAUSSIAN_RENDER_MODE_SHARED_TEMPLATE,
):
    gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
    template_gaussians = load_gaussian_template(
        gs_path,
        sh_degree=sh_degree,
        opacity_threshold=opacity_threshold,
    )
    instance_offsets = _resolve_instance_offsets(
        number_of_instance=number_of_instance,
        template_gaussians=template_gaussians,
        offset_step=offset_step,
        instance_offsets=instance_offsets,
    )
    shared_state = SharedGaussianBatchState(template_gaussians, instance_offsets)
    if gaussian_render_mode == GAUSSIAN_RENDER_MODE_DUPLICATED:
        return shared_state, RenderCompatibleGaussianView(shared_state)
    return shared_state, shared_state
