"""Plot the learning rate schedules of the runoff and routing optimizers over training (saves lr_schedule.png)."""

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
runoff_lr = 2e-5
routing_lr = 2e-5
runoff_freeze_epochs = 5
warmup_epochs = 5
cycle_epochs = 20
max_epochs = 80
steps_per_epoch = 10

# ── Derived ───────────────────────────────────────────────────────────────────
stepping_batches = max_epochs * steps_per_epoch
cycle_steps = min(cycle_epochs * steps_per_epoch, stepping_batches)
freeze_steps = runoff_freeze_epochs * steps_per_epoch
warmup_pct_start = min(warmup_epochs, cycle_epochs - 1) / cycle_epochs


# ── Dummy optimizers ──────────────────────────────────────────────────────────
def dummy_params() -> list[torch.nn.Parameter]:
    """Return a single dummy parameter, so an optimizer can be created without a model."""
    return [torch.nn.Parameter(torch.zeros(1))]


runoff_optimizer = AdamW(dummy_params(), lr=runoff_lr)
routing_optimizer = AdamW(dummy_params(), lr=routing_lr)

# ── Routing: cosine → linear tail ─────────────────────────────────────────────
remaining_routing = stepping_batches - cycle_steps
routing_schedulers = [
    CosineAnnealingLR(routing_optimizer, T_max=cycle_steps, eta_min=routing_lr / 25),
]
routing_milestones = []
if remaining_routing > 0:
    routing_schedulers.append(
        LinearLR(routing_optimizer, start_factor=1 / 25, end_factor=1e-6, total_iters=remaining_routing)
    )
    routing_milestones.append(cycle_steps)
routing_scheduler = SequentialLR(routing_optimizer, schedulers=routing_schedulers, milestones=routing_milestones)

# ── Runoff: freeze → OneCycleLR → linear tail ─────────────────────────────────
runoff_total_steps = stepping_batches - freeze_steps
runoff_cycle_steps = min(cycle_steps, runoff_total_steps)
remaining_runoff = runoff_total_steps - runoff_cycle_steps
runoff_schedulers = [
    ConstantLR(runoff_optimizer, factor=0.0, total_iters=freeze_steps),
    OneCycleLR(
        optimizer=runoff_optimizer,
        max_lr=runoff_lr,
        total_steps=runoff_cycle_steps,
        pct_start=warmup_pct_start,
        anneal_strategy="cos",
        final_div_factor=1.0,
    ),
]
runoff_milestones = [freeze_steps]
if remaining_runoff > 0:
    runoff_schedulers.append(LinearLR(runoff_optimizer, start_factor=1.0, end_factor=0.1, total_iters=remaining_runoff))
    runoff_milestones.append(freeze_steps + runoff_cycle_steps)
runoff_scheduler = SequentialLR(runoff_optimizer, schedulers=runoff_schedulers, milestones=runoff_milestones)

# ── Simulate ──────────────────────────────────────────────────────────────────
routing_lrs, runoff_lrs = [], []
for _ in range(stepping_batches):
    routing_lrs.append(routing_optimizer.param_groups[0]["lr"])
    runoff_lrs.append(runoff_optimizer.param_groups[0]["lr"])
    routing_scheduler.step()
    runoff_scheduler.step()

epochs = [s / steps_per_epoch for s in range(stepping_batches)]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, routing_lrs, label="routing LR")
ax.plot(epochs, runoff_lrs, label="runoff LR")
for e, label in [(runoff_freeze_epochs, "unfreeze"), (cycle_epochs, "cycle end")]:
    ax.axvline(e, ls="--", color="gray", alpha=0.6)
    ax.text(e + 0.3, max(routing_lrs) * 0.95, label, fontsize=8, color="gray")
ax.set_xlabel("epoch")
ax.set_ylabel("LR")
ax.legend()
ax.set_title("LR schedule")
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=150)
plt.show()
