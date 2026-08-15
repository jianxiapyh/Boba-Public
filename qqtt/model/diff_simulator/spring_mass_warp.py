import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


SIM_FORCE_MODE_GATHER = "gather"
SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC = "template_state_batched_atomic"
SIM_FORCE_MODES = (
    SIM_FORCE_MODE_GATHER,
    SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC,
)


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()


    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )



@wp.kernel
def eval_springs_single_instance_atomic(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    inv_rest_lengths: wp.array(dtype=float),
    spring_Y_clamped: wp.array(dtype=float),
    dashpot_damping: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    idx1 = springs[tid][0]
    idx2 = springs[tid][1]

    if idx1 >= num_object_points:
        x1 = control_x[idx1 - num_object_points]
        v1 = control_v[idx1 - num_object_points]
    else:
        x1 = x[idx1]
        v1 = v[idx1]

    if idx2 >= num_object_points:
        x2 = control_x[idx2 - num_object_points]
        v2 = control_v[idx2 - num_object_points]
    else:
        x2 = x[idx2]
        v2 = v[idx2]

    inv_rest = inv_rest_lengths[tid]
    y = spring_Y_clamped[tid]

    dis = x2 - x1
    dis_len = wp.length(dis)
    d = dis / wp.max(dis_len, 1e-6)

    spring_force = y * (dis_len * inv_rest - 1.0) * d
    v_rel = wp.dot(v2 - v1, d)
    dashpot_forces = dashpot_damping * v_rel * d
    overall_force = spring_force + dashpot_forces

    if idx1 < num_object_points:
        wp.atomic_add(f, idx1, overall_force)
    if idx2 < num_object_points:
        wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def eval_springs_batched_template_state_atomic(
    # batched state
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),

    # BASE template springs (size = n_springs)
    springs: wp.array(dtype=wp.vec2i),
    inv_rest_lengths: wp.array(dtype=float),
    spring_Y_clamped: wp.array(dtype=float),
    dashpot_damping: float,

    # sizes
    object_massnode_single: int,
    controller_massnode_single: int,
    n_springs: int,

    # output: per-instance object-node forces
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // n_springs
    local_idx = tid - inst * n_springs

    local_idx1 = springs[local_idx][0]
    local_idx2 = springs[local_idx][1]

    if local_idx1 >= object_massnode_single:
        ctrl1 = inst * controller_massnode_single + (local_idx1 - object_massnode_single)
        x1 = control_x[ctrl1]
        v1 = control_v[ctrl1]
    else:
        global_idx1 = inst * object_massnode_single + local_idx1
        x1 = x[global_idx1]
        v1 = v[global_idx1]

    if local_idx2 >= object_massnode_single:
        ctrl2 = inst * controller_massnode_single + (local_idx2 - object_massnode_single)
        x2 = control_x[ctrl2]
        v2 = control_v[ctrl2]
    else:
        global_idx2 = inst * object_massnode_single + local_idx2
        x2 = x[global_idx2]
        v2 = v[global_idx2]

    inv_rest = inv_rest_lengths[local_idx]
    y = spring_Y_clamped[local_idx]

    dis = x2 - x1
    dis_len = wp.length(dis)
    d = dis / wp.max(dis_len, 1e-6)

    spring_force = y * (dis_len * inv_rest - 1.0) * d
    v_rel = wp.dot(v2 - v1, d)
    dashpot_forces = dashpot_damping * v_rel * d
    overall_force = spring_force + dashpot_forces

    if local_idx1 < object_massnode_single:
        wp.atomic_add(f, inst * object_massnode_single + local_idx1, overall_force)
    if local_idx2 < object_massnode_single:
        wp.atomic_sub(f, inst * object_massnode_single + local_idx2, overall_force)


@wp.kernel
def eval_springs_batched_compute_all_base(
    # batched state
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),

    # BASE template springs (size = n_springs)
    springs: wp.array(dtype=wp.vec2i),
    inv_rest_lengths: wp.array(dtype=float),
    spring_Y_clamped: wp.array(dtype=float),
    dashpot_damping: float,

    # sizes
    object_massnode_single: int,
    controller_massnode_single: int,
    n_springs: int,
    number_of_instance: int,

    # output: per-instance per-base-spring force (size = number_of_instance * n_springs)
    spring_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inst = tid // n_springs
    local_idx = tid - inst * n_springs  # BASE spring id in [0, n_springs)

    y = spring_Y_clamped[local_idx]

    local_idx1 = springs[local_idx][0]
    local_idx2 = springs[local_idx][1]

    # endpoint 1
    if local_idx1 >= object_massnode_single:
        ctrl1 = inst * controller_massnode_single + (local_idx1 - object_massnode_single)
        x1 = control_x[ctrl1]
        v1 = control_v[ctrl1]
    else:
        global_idx1 = inst * object_massnode_single + local_idx1
        x1 = x[global_idx1]
        v1 = v[global_idx1]

    # endpoint 2
    if local_idx2 >= object_massnode_single:
        ctrl2 = inst * controller_massnode_single + (local_idx2 - object_massnode_single)
        x2 = control_x[ctrl2]
        v2 = control_v[ctrl2]
    else:
        global_idx2 = inst * object_massnode_single + local_idx2
        x2 = x[global_idx2]
        v2 = v[global_idx2]

    inv_rest = inv_rest_lengths[local_idx]

    dis = x2 - x1
    dis_len = wp.length(dis)
    d = dis / wp.max(dis_len, 1e-6)

    spring_force = y * (dis_len * inv_rest - 1.0) * d
    v_rel = wp.dot(v2 - v1, d)
    dashpot_forces = dashpot_damping * v_rel * d

    #[spring][instance]
    # out_idx = local_idx * number_of_instance + inst
    # spring_out[out_idx] = spring_force + dashpot_forces
    #instance major [inst][spring]
    spring_out[tid] = spring_force + dashpot_forces

@wp.kernel
def reduce_object_force_from_signed_incidence_instance(
    signed_incidence_map: wp.array(dtype=wp.int32),
    spring_force_lookup: wp.array(dtype=wp.vec3),

    object_massnode_single: int,
    max_incident_springs: int,
    n_springs: int,

    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # inst-major mapping (inst constant for a chunk of threads)
    inst = tid // object_massnode_single
    obj_node = tid - inst * object_massnode_single

    acc = wp.vec3(0.0, 0.0, 0.0)
    base_row = obj_node * max_incident_springs

    for incident_idx in range(max_incident_springs):
        sid = signed_incidence_map[base_row + incident_idx]
        if sid != 0:
            sign = 1.0
            if sid < 0:
                sign = -1.0
                sid = -sid
            base_idx = sid - 1

            force = spring_force_lookup[inst * n_springs + base_idx]  # <-- instance-major
            acc = acc + sign * force

    f[inst * object_massnode_single + obj_node] = acc


#updated since masses are shared
@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    local_idx = 0

    inst = tid // object_massnode_single
    local_idx = tid - inst * object_massnode_single

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[local_idx]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
    mass_i: float, 
    offset: int, 
):
    x1 = x[i]
    v1 = v[i]

    #pyh added 
    m1 = mass_i
    mask1 = masks[i-offset]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]

        #pyh added
        local_idx2 = index - offset
        m2 = masses[local_idx2]
        mask2 = masks[local_idx2]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


#pyh this is getting updated since resting_collision_pair is single instance now
@wp.kernel(enable_backward=False)
def update_potential_collision_restmap(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
    object_massnode_single: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    #pyh added
    inst_i = i//object_massnode_single
    local_i = i - inst_i * object_massnode_single

    x1 = x[i]
    #pyh changed
    mask1 = masks[local_i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)

    for neighbor_index in neighbors:
        if neighbor_index != i:
            inst_neighbor = neighbor_index // object_massnode_single
            #if they are not from same instance we also skip 
            if inst_neighbor != inst_i:
                continue
            local_neighbor = neighbor_index - inst_neighbor * object_massnode_single

            if resting_collision_pairs[local_i][local_neighbor] == True or resting_collision_pairs[local_neighbor][local_i] == True:
                continue    
            x2 = x[neighbor_index]        
            mask2 = masks[local_neighbor]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = neighbor_index
                collision_number[i] += 1

@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    local_idx = 0
    inst = tid // object_massnode_single
    offset = inst * object_massnode_single
    local_idx = tid - offset

    v1 = v[tid]
    m1 = masses[local_idx]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
        m1,
        offset,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1

#pyh this is changing to batched version
@wp.kernel(enable_backward=False)
def build_resting_collision_pairs(
    x: wp.array(dtype=wp.vec3),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index < i:
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)

@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1

class SpringMassSystemWarp:
    def __init__(
        self,
        base_springs,
        base_rest_lengths,
        init_masses,
        init_masks,
        signed_incidence_map,
        max_incident_springs,
        init_vertices,
        init_velocities,
        dt, 
        num_substeps, 
        dashpot_damping,  
        drag_damping, 
        collision_dist, 
        reverse_z, 
        spring_Y_min, 
        spring_Y_max, 
        self_collision, 
        #updated variable
        collide_elas, 
        collide_fric, 
        collide_object_elas, 
        collide_object_fric, 
        spring_Y,
        object_massnodes_total,
        object_massnodes_single,
        controller_massnodes_single, 
        controller_rest_location,
        number_of_instance,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )

        self.sim_force_mode = sim_force_mode
        self.use_gather_solver = (
            number_of_instance > 1 and sim_force_mode == SIM_FORCE_MODE_GATHER
        )
        self.use_template_state_batched_atomic = (
            sim_force_mode == SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC
        )
        if self.use_gather_solver and signed_incidence_map is None:
            raise ValueError("signed_incidence_map is required when number_of_instance > 1")

        self.n_springs_single = int(base_springs.shape[0])
        self.n_springs_batched = self.n_springs_single * number_of_instance

        self.wp_template_springs = wp.from_torch(
            base_springs, dtype=wp.vec2i, requires_grad=False
        )
        
        self.wp_inv_template_rest_length = wp.from_torch(
            1.0 / base_rest_lengths, dtype=wp.float32, requires_grad=False
        )

        self.wp_spring_force_lookup = None
        self.wp_signed_incidence_map = None
        if self.use_gather_solver:
            # Temporary per-spring force buffer used by the two-stage gather assembly.
            self.wp_spring_force_lookup = wp.zeros(
                (self.n_springs_batched,), dtype=wp.vec3, requires_grad=False
            )
            self.wp_signed_incidence_map = wp.from_torch(
                signed_incidence_map.to(dtype=torch.int32).contiguous(),
                dtype=wp.int32,
                requires_grad=False,
            )

        self.wp_masses = wp.from_torch(
            init_masses, dtype=wp.float32, requires_grad=False
        )
        self.wp_masks = None #only useful when self-collision is on
        if self_collision:
            if init_masks is None:
                print(f"Using default masks for self-collision")
                default_masks = torch.arange(
                    object_massnodes_single, dtype=torch.int32, device=self.device
                )            
                self.wp_masks = wp.from_torch(default_masks, dtype=wp.int32, requires_grad=False)
            else:
                print(f"Using provided masks for self-collision")
                assert init_masks.shape[0] == object_massnodes_single, "init_masks shape mismatch"
                self.wp_masks = wp.from_torch(
                    init_masks, dtype=wp.int32, requires_grad=False
                )

        #a large vector of all object mass nodes from all instances
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:object_massnodes_total].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )

        #ones doesnt change 
        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.collision_dist = collision_dist
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max
        # variable for collision detection
        self.object_collision_flag = 0
        self.resting_collision_pairs = None
        self.wp_single_x = None
        self.collision_grid = None
        self.wp_collision_indices = None
        self.wp_collision_number = None
        
        if self_collision:
            self.object_collision_flag = 1
            self.resting_collision_pairs = wp.zeros((object_massnodes_single, object_massnodes_single),dtype=wp.bool, requires_grad=False)
        
            self.wp_single_x = wp.empty(
                shape=(object_massnodes_single, ),
                dtype=wp.vec3,
                device=cfg.device,
                requires_grad=False,
            )
            
            self.collision_grid = wp.HashGrid(128, 128, 128)

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        self.wp_collide_elas = wp.from_torch(
            collide_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            collide_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            collide_object_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            collide_object_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )

        spring_Y_temp = spring_Y.to(device=self.device, dtype=torch.float32).contiguous()
        spring_Y_clamped = spring_Y_temp.clamp(min=self.spring_Y_min, max=self.spring_Y_max)

        self.wp_spring_Y_clamped = wp.from_torch(
            spring_Y_clamped, dtype=wp.float32, requires_grad=False
        )

        assert spring_Y_clamped.numel() == self.n_springs_single
        
        self.object_massnode_single = object_massnodes_single
        self.object_massnode_total = object_massnodes_total
        self.controller_massnode_single = controller_massnodes_single
        self.number_of_instance = number_of_instance
        self.max_incident_springs = int(max_incident_springs)

        self.wp_init_velocities = None
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros(
                shape=(self.object_massnode_total, ),
                dtype=wp.vec3,
                device=cfg.device,
                requires_grad=False)
        else:
            assert init_velocities.shape[0] == object_massnodes_total, "init_velocities shape mismatch"
            self.wp_init_velocities = wp.from_torch(
                init_velocities.contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.wp_original_control_point = wp.from_torch(
            controller_rest_location.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.wp_target_control_point = wp.from_torch(
            controller_rest_location.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.num_controller_points = controller_rest_location.shape[0]
  
        # Preallocating the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_vertices, self.num_controller_points)
            self.wp_states.append(state)

    
    #pyh creating cuda graph requires all scalar variable to be set, that's not possible in batched 
    #so we are moving it out
    def create_cuda_graph(self):
        # Create the CUDA graph to acclerate
        with wp.ScopedCapture() as forward_capture:
            self.step()   
        self.forward_graph = forward_capture.graph

    # Create the rest map for self-collision in frame 0
    #pyh updated for batched version
    def create_resting_case(self):
        #we only need build restmap for one instance to be shared but warp does not allow slicing 
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_single,
            inputs=[self.wp_states[0].wp_x],
            outputs=[self.wp_single_x],
        )

        self.collision_grid.build(self.wp_single_x, self.collision_dist * 5.0)
        print("create resting case implementation (doesn't mean its in use check update_collision_graph)")
        wp.launch(
            build_resting_collision_pairs,
            dim=self.object_massnode_single,
            inputs=[
                self.wp_single_x,
                self.collision_dist,
                self.collision_grid.id,
                ],
            outputs=[self.resting_collision_pairs],            
        )  
    
    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v):
        assert (
            self.object_massnode_total == wp_x.shape[0]
            and self.object_massnode_total == self.wp_states[0].wp_x.shape[0]
        )

        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_x],
            outputs=[self.wp_states[0].wp_x],
        )
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_v],
            outputs=[self.wp_states[0].wp_v],
        )

    def update_collision_graph(self):
        #pyh build a big hash grid over all instances are ok as long as the offset is big
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()

        wp.launch(
          update_potential_collision_restmap,
          dim=self.object_massnode_total,
          inputs=[
              self.wp_states[0].wp_x,
              self.wp_masks,
              self.collision_dist,
              self.collision_grid.id,
              self.resting_collision_pairs,
              self.object_massnode_single,
          ],
          outputs=[self.wp_collision_indices, self.wp_collision_number],
        )


    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()

            # Set the control point
            wp.launch(
                set_control_points,
                dim=self.num_controller_points,
                inputs=[
                    self.num_substeps,
                    self.wp_original_control_point,
                    self.wp_target_control_point,
                    i,
                ],
                outputs=[self.wp_states[i].wp_control_x],
            )
            
            if self.use_gather_solver:
                wp.launch(
                    kernel=eval_springs_batched_compute_all_base,
                    dim=self.n_springs_batched,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v,
                        self.wp_states[i].wp_control_x,
                        self.wp_states[i].wp_control_v,
                        self.wp_template_springs,
                        self.wp_inv_template_rest_length,
                        self.wp_spring_Y_clamped,
                        self.dashpot_damping,
                        self.object_massnode_single,
                        self.controller_massnode_single,
                        self.n_springs_single,
                        self.number_of_instance,
                    ],
                    outputs=[self.wp_spring_force_lookup],
                )

                wp.launch(
                    kernel=reduce_object_force_from_signed_incidence_instance,
                    dim=self.number_of_instance * self.object_massnode_single,
                    inputs=[
                        self.wp_signed_incidence_map,
                        self.wp_spring_force_lookup,
                        self.object_massnode_single,
                        self.max_incident_springs,
                        self.n_springs_single,
                    ],
                    outputs=[self.wp_states[i].wp_vertice_forces],
                )
            elif self.use_template_state_batched_atomic:
                wp.launch(
                    kernel=eval_springs_batched_template_state_atomic,
                    dim=self.n_springs_batched,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v,
                        self.wp_states[i].wp_control_x,
                        self.wp_states[i].wp_control_v,
                        self.wp_template_springs,
                        self.wp_inv_template_rest_length,
                        self.wp_spring_Y_clamped,
                        self.dashpot_damping,
                        self.object_massnode_single,
                        self.controller_massnode_single,
                        self.n_springs_single,
                    ],
                    outputs=[self.wp_states[i].wp_vertice_forces],
                )
            else:
                wp.launch(
                    kernel=eval_springs_single_instance_atomic,
                    dim=self.n_springs_single,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v,
                        self.wp_states[i].wp_control_x,
                        self.wp_states[i].wp_control_v,
                        self.object_massnode_single,
                        self.wp_template_springs,
                        self.wp_inv_template_rest_length,
                        self.wp_spring_Y_clamped,
                        self.dashpot_damping,
                    ],
                    outputs=[self.wp_states[i].wp_vertice_forces],
                )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    #shared
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                    #pyh added
                    self.object_massnode_single,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.object_massnode_total,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        #shared
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                        #added
                        self.object_massnode_single,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )
        
