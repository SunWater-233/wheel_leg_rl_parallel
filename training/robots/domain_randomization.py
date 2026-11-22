"""Sample one fixed physical domain for each Genesis environment."""

import torch


def randomize_dynamics(env, settings):
    robot = env.robot
    template = env.commands
    count = env.num_envs
    nominal = torch.rand(count, device=env.device) < settings["nominal_probability"]

    friction = template.new_tensor(settings["contact_friction"])
    buckets = torch.randint(len(friction), (count,), device=env.device)
    baseline = env.cfg["friction"]
    sampled_friction = torch.where(nominal, baseline, friction[buckets])
    ratio = sampled_friction / baseline
    # Genesis uses max(wheel, terrain) for a contact pair.
    for entity in (robot, *env.terrain_contacts):
        entity.set_friction_ratio(ratio[:, None].expand(-1, entity.n_links))

    low, high = settings["body_mass_scale"]
    scale = torch.where(nominal, 1.0, template.new_empty(count).uniform_(low, high))
    base = [robot.get_link("base_link").idx_local]
    mass = robot.get_links_mass(base)
    center = robot.get_links_COM(base)
    inertia = robot.get_links_inertia(base)
    offsets = template.new_tensor(settings["load_offsets"])
    load = offsets[torch.randint(len(offsets), (count,), device=env.device)]
    # Tie the COM shift to the signed change in mass at the sampled payload position.
    center_shift = load * ((scale - 1.0) / scale)[:, None]
    robot.set_links_mass(mass * scale[:, None], base)
    robot.set_links_COM(center + center_shift[:, None, :], base)
    robot.set_links_inertia(inertia * scale[:, None, None, None], base)

    for name, dofs in (("hips", env.leg_dofs), ("passive", env.passive_dofs),
                       ("springs", env.spring_dofs), ("wheels", env.wheel_dofs)):
        group = settings[name]
        shape = (count, len(dofs))
        friction_delta = template.new_empty(shape).uniform_(*group["frictionloss_delta"])
        damping_delta = template.new_empty(shape).uniform_(*group["damping_delta"])
        frictionloss = robot.get_dofs_frictionloss(dofs) + friction_delta.masked_fill(nominal[:, None], 0.0)
        damping = robot.get_dofs_damping(dofs) + damping_delta.masked_fill(nominal[:, None], 0.0)
        robot.set_dofs_frictionloss(frictionloss.clamp_min(0.0), dofs)
        robot.set_dofs_damping(damping.clamp_min(0.0), dofs)

    counts = [int((sampled_friction == mu).sum()) for mu in friction]
    print(f"Domain randomization: {int(nominal.sum())}/{count} nominal; "
          f"friction bucket counts {dict(zip(settings['contact_friction'], counts))}; "
          f"base mass scale {float(scale.min()):.3f}-{float(scale.max()):.3f}; "
          f"max base COM shift {float(center_shift.norm(dim=1).max()) * 1000:.1f} mm", flush=True)
