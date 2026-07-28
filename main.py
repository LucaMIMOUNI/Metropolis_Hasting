import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Basic surface function to test optimisation algorithms
def surface(x, y):
    return -(x**2 + y**2 + 10 * np.sin(x) * np.cos(y))


def metropolis_hastings(initial_state, iterations, proposal_std):
    current_state = initial_state
    samples = [current_state]
    step_records = []

    for _ in range(iterations):
        proposed_state = current_state + np.random.normal(0, proposal_std, size=current_state.shape)

        current_prob = np.exp(surface(*current_state))
        proposed_prob = np.exp(surface(*proposed_state))
        acceptance_prob = min(1.0, proposed_prob / current_prob)
        accepted = np.random.rand() < acceptance_prob

        if accepted:
            current_state = proposed_state

        samples.append(current_state)
        step_records.append(
            {
                "current_prob": current_prob,
                "proposed_prob": proposed_prob,
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
            }
        )

    return np.array(samples), step_records


def animate_MH(ax, samples, step_records, z_samples, frames_per_step=3, panel_ax=None):
    trail, = ax.plot([], [], [], color="red", linewidth=1.5)
    point = ax.scatter([], [], [], color="red", s=60, edgecolor="white")
    start_point = ax.scatter(
        samples[0, 0],
        samples[0, 1],
        z_samples[0],
        color="lime",
        s=180,
        edgecolors="black",
        linewidths=1.2,
        depthshade=False,
    )

    trail_x = [samples[0, 0]]
    trail_y = [samples[0, 1]]
    trail_z = [z_samples[0]]
    total_frames = (len(samples) - 1) * frames_per_step + 1

    if panel_ax is not None:
        panel_ax.set_axis_off()
        panel_ax.text(
            0.02,
            0.96,
            "Acceptance check",
            transform=panel_ax.transAxes,
            fontsize=13,
            fontweight="semibold",
            va="top",
        )
        panel_ax.text(
            0.02,
            0.86,
            r"$\mathrm{acceptance\_prob} = \min\left(1.0, \frac{\mathrm{proposed\_prob}}{\mathrm{current\_prob}}\right)$",
            transform=panel_ax.transAxes,
            fontsize=10.5,
            va="top",
        )
        equation_text = panel_ax.text(
            0.02,
            0.62,
            "",
            transform=panel_ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace",
        )
        status_text = panel_ax.text(
            0.5,
            0.18,
            "",
            transform=panel_ax.transAxes,
            fontsize=16,
            fontweight="semibold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f3f3", edgecolor="#cccccc"),
        )
    else:
        equation_text = None
        status_text = None

    def update(frame):
        if frame < total_frames - 1:
            step_index, step_frame = divmod(frame, frames_per_step)
            start = samples[step_index]
            end = samples[step_index + 1]
            record = step_records[step_index]
            alpha = step_frame / frames_per_step
            eased_alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            position = start + eased_alpha * (end - start)
            z_position = surface(position[0], position[1])
            trail_x.append(position[0])
            trail_y.append(position[1])
            trail_z.append(z_position)
        else:
            position = samples[-1]
            z_position = z_samples[-1]
            record = step_records[-1]

        trail.set_data_3d(trail_x, trail_y, trail_z)
        point._offsets3d = ([position[0]], [position[1]], [z_position])

        if equation_text is not None and status_text is not None:
            equation_text.set_text(
                f"= min(1.0, {record['proposed_prob']:.3e} / {record['current_prob']:.3e})\n"
                f"= {record['acceptance_prob']:.3f}"
            )
            status_text.set_text("accept" if record["accepted"] else "stay")
            status_text.set_color("seagreen" if record["accepted"] else "crimson")

        return trail, point, start_point

    ax.figure._animation = FuncAnimation(ax.figure, update, frames=total_frames, interval=5, repeat=False)
    return start_point


def main():
    # MH init
    initial_state = np.array([-4.5, 4.5])
    iterations = 400
    proposal_std = 0.5

    # Plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = surface(X, Y)

    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(left=0.06, right=0.72, top=0.88, bottom=0.08)
    ax = fig.add_subplot(111, projection="3d")
    panel_ax = fig.add_axes([0.75, 0.14, 0.23, 0.68])
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    fig.suptitle("Metropolis-Hastings Sampling", fontsize=16, fontweight="semibold", y=0.98)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)

    # MH
    samples, step_records = metropolis_hastings(initial_state, iterations, proposal_std)
    z_samples = surface(samples[:, 0], samples[:, 1])

    # Animation
    start_point = animate_MH(ax, samples, step_records, z_samples, panel_ax=panel_ax)
    ax.legend([start_point], ["Start"], loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()