def linear_decay(step, eps_start=1.0, eps_end=0.05, decay_steps=10_000):
    if step >= decay_steps:
        return eps_end
    else:
        return eps_start - (eps_start - eps_end) * (step / decay_steps)