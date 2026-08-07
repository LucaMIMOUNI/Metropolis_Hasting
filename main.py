import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

FRAMES_PER_STEP = 4   # interpolation frames between two chain states
INTERVAL_MS = 20      # timer period at x1
ORBIT_FRAMES = 720    # frames for one full turn of the camera (~14 s)
Z_LIFT = 0.8          # raise the chain above the surface so it stays visible
TRAIL = "#343a40"     # the chain, neutral so the colour code carries the meaning
ACCEPTED = "#0353a4"  # proposals the acceptance test kept
REJECTED = "#c1121f"  # proposals it turned down
PLAY_FILL = "#2a6f97"  # play button, filled while the animation is waiting on you
PLAY_HOVER = "#22587a"


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
                "proposed_state": proposed_state,
                "current_prob": current_prob,
                "proposed_prob": proposed_prob,
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
            }
        )

    return np.array(samples), step_records


def build_panel(panel_ax):
    panel_ax.set_axis_off()
    panel_ax.text(0, 1.0, "Acceptance test", transform=panel_ax.transAxes,
                  fontsize=12, fontweight="semibold", va="top")
    panel_ax.text(0, 0.90, r"$\alpha = \min\left(1,\; \dfrac{\pi(y)}{\pi(x)}\right)$",
                  transform=panel_ax.transAxes, fontsize=13, va="top")
    equation_text = panel_ax.text(0, 0.72, "", transform=panel_ax.transAxes,
                                  fontsize=9.5, va="top", family="monospace")
    status_text = panel_ax.text(0, 0.52, "", transform=panel_ax.transAxes,
                                fontsize=12, fontweight="semibold", va="top", family="monospace")
    panel_ax.plot([0, 1], [0.44, 0.44], transform=panel_ax.transAxes, color="#cccccc", linewidth=0.8)
    stats_text = panel_ax.text(0, 0.38, "", transform=panel_ax.transAxes,
                               fontsize=9.5, va="top", family="monospace")
    return equation_text, status_text, stats_text


def animate_MH(fig, ax, samples, step_records, z_samples, panel_ax):
    equation_text, status_text, stats_text = build_panel(panel_ax)

    lifted_z = z_samples + Z_LIFT
    trail, = ax.plot([], [], [], color=TRAIL, linewidth=1.1, alpha=0.85, zorder=3)

    # Every proposal is marked, colour-coded by the outcome of the acceptance test.
    # Accepted ones sit on the path; rejected ones show where the chain declined to go.
    rejected_marks, = ax.plot([], [], [], linestyle="none", marker="x", markersize=4.5,
                              markeredgewidth=1.0, color=REJECTED, alpha=0.75, zorder=2,
                              label="rejected proposal")
    accepted_marks, = ax.plot([], [], [], linestyle="none", marker="x", markersize=5,
                              markeredgewidth=1.2, color=ACCEPTED, zorder=3,
                              label="accepted proposal")
    head, = ax.plot([], [], [], linestyle="none", marker="o", markersize=7, color=TRAIL,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=4,
                    label="current state")
    ax.plot([samples[0, 0]], [samples[0, 1]], [lifted_z[0]], linestyle="none", marker="o",
            markersize=7, color="white", markeredgecolor=TRAIL, markeredgewidth=1.5,
            zorder=4, label="start")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    n_steps = len(samples) - 1
    total_frames = n_steps * FRAMES_PER_STEP
    accepted = np.array([r["accepted"] for r in step_records])
    accepted_cum = np.cumsum(accepted)
    proposals = np.array([r["proposed_state"] for r in step_records])
    proposal_z = surface(proposals[:, 0], proposals[:, 1]) + Z_LIFT
    accepted_idx = np.flatnonzero(accepted)
    rejected_idx = np.flatnonzero(~accepted)
    base_elev, base_azim = ax.elev, ax.azim
    # Starts paused: the viewer presses play. `orbit` counts frames into a full turn.
    state = {"frame": 0.0, "speed": 1.0, "playing": False, "orbit": None}

    def draw(frame):
        frame = min(frame, total_frames - 1e-6)
        step_index = int(frame // FRAMES_PER_STEP)
        t = (frame % FRAMES_PER_STEP) / FRAMES_PER_STEP
        eased = 0.5 - 0.5 * np.cos(np.pi * t)

        start, end = samples[step_index], samples[step_index + 1]
        position = start + eased * (end - start)
        z_position = surface(position[0], position[1]) + Z_LIFT

        # Path = the accepted states so far, with the interpolated head appended.
        px = np.append(samples[: step_index + 1, 0], position[0])
        py = np.append(samples[: step_index + 1, 1], position[1])
        pz = np.append(lifted_z[: step_index + 1], z_position)
        trail.set_data_3d(px, py, pz)
        head.set_data_3d([position[0]], [position[1]], [z_position])

        # Proposals drawn so far, i.e. those of the steps already played.
        for marks, idx in ((accepted_marks, accepted_idx), (rejected_marks, rejected_idx)):
            shown = idx[: np.searchsorted(idx, step_index, side="right")]
            marks.set_data_3d(proposals[shown, 0], proposals[shown, 1], proposal_z[shown])

        record = step_records[step_index]
        equation_text.set_text(
            f"pi(y) = {record['proposed_prob']:.3e}\n"
            f"pi(x) = {record['current_prob']:.3e}\n"
            f"alpha = {record['acceptance_prob']:.3f}"
        )
        status_text.set_text("accept" if record["accepted"] else "stay")
        status_text.set_color(ACCEPTED if record["accepted"] else REJECTED)
        stats_text.set_text(
            f"step        {step_index + 1:>4d} / {n_steps}\n"
            f"acceptance  {accepted_cum[step_index] / (step_index + 1):>7.1%}\n"
            f"position    ({position[0]:+.2f}, {position[1]:+.2f})\n"
            f"f(x, y)     {z_position - Z_LIFT:+8.2f}"
        )

    def orbit_step():
        """Advance the camera along a smooth full turn, then restore the view."""
        state["orbit"] += 1
        progress = state["orbit"] / ORBIT_FRAMES
        if progress >= 1.0:
            stop_orbit()
        else:
            eased = progress * progress * (3 - 2 * progress)  # gentle start and finish
            ax.view_init(elev=base_elev, azim=base_azim + 360.0 * eased)

    def update(_):
        if state["playing"]:
            state["frame"] += state["speed"]
            if state["frame"] >= total_frames - 1:
                state["frame"] = total_frames - 1
                state["playing"] = False
                play_button.label.set_text("replay")
                style_play()
        if state["orbit"] is not None:
            orbit_step()
        draw(state["frame"])
        return ()

    # --- controls ------------------------------------------------------------
    def style_button(button, active=False):
        button.color = "#dfe4ea" if active else "#f4f5f7"
        button.hovercolor = "#d0d7de" if active else "#e8eaed"
        button.ax.set_facecolor(button.color)
        button.label.set_fontsize(10)
        button.label.set_fontweight("semibold" if active else "normal")

    def style_play():
        """Filled while the animation is waiting on you, plain while it runs."""
        idle = not state["playing"]
        play_button.color = PLAY_FILL if idle else "#f4f5f7"
        play_button.hovercolor = PLAY_HOVER if idle else "#e8eaed"
        play_button.ax.set_facecolor(play_button.color)
        play_button.label.set_color("white" if idle else "black")
        play_button.label.set_fontweight("semibold" if idle else "normal")

    def add_button(rect, label):
        button = Button(fig.add_axes(rect), label, color="#f4f5f7", hovercolor="#e8eaed")
        style_button(button)
        return button

    play_button = add_button([0.08, 0.035, 0.09, 0.045], "play")
    speed_buttons = {s: add_button([0.19 + i * 0.055, 0.035, 0.045, 0.045], f"x{s}")
                     for i, s in enumerate((1, 2, 5, 10))}
    orbit_button = add_button([0.43, 0.035, 0.085, 0.045], "orbit 360°")
    style_button(speed_buttons[1], active=True)
    style_play()

    def toggle_play(_event=None):
        if not state["playing"] and state["frame"] >= total_frames - 1:
            state["frame"] = 0.0  # finished -> replay from the top
        state["playing"] = not state["playing"]
        play_button.label.set_text("pause" if state["playing"] else "play")
        style_play()
        fig.canvas.draw_idle()

    def stop_orbit():
        state["orbit"] = None
        ax.view_init(elev=base_elev, azim=base_azim)
        orbit_button.label.set_text("orbit 360°")
        style_button(orbit_button, active=False)

    def toggle_orbit(_event=None):
        if state["orbit"] is None:
            state["orbit"] = 0
            orbit_button.label.set_text("stop orbit")
            style_button(orbit_button, active=True)
        else:
            stop_orbit()
        fig.canvas.draw_idle()

    def set_speed(value):
        def handler(_event=None):
            state["speed"] = float(value)
            for s, button in speed_buttons.items():
                style_button(button, active=(s == value))
            fig.canvas.draw_idle()
        return handler

    play_button.on_clicked(toggle_play)
    orbit_button.on_clicked(toggle_orbit)
    for s, button in speed_buttons.items():
        button.on_clicked(set_speed(s))

    def on_key(event):
        if event.key == " ":
            toggle_play()
        elif event.key in ("1", "2", "5"):
            set_speed(int(event.key))()
        elif event.key == "0":
            set_speed(10)()
        elif event.key == "r":
            toggle_orbit()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(0.08, 0.008, "space = play/pause     1 / 2 / 5 / 0 = speed     r = orbit",
             fontsize=8, color="#888888")

    # Buttons stop responding if they are garbage collected -> keep them alive.
    fig._mh_controls = {"play": play_button, "speeds": speed_buttons,
                        "orbit": orbit_button, "state": state}

    draw(0.0)
    fig._animation = FuncAnimation(fig, update, frames=itertools.count, interval=INTERVAL_MS,
                                   blit=False, repeat=False, cache_frame_data=False)


def main():
    # MH init
    initial_state = np.array([-4.5, 4.5])
    iterations = 400
    proposal_std = 0.5

    # Plot
    x = np.linspace(-5, 5, 120)
    y = np.linspace(-5, 5, 120)
    X, Y = np.meshgrid(x, y)
    Z = surface(X, Y)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.02, 0.10, 0.66, 0.80], projection="3d")
    panel_ax = fig.add_axes([0.73, 0.30, 0.24, 0.50])

    # Honour zorder literally: with matplotlib's per-artist depth sort the surface is
    # drawn over the whole chain and the walk disappears into it.
    ax.computed_zorder = False
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.75, linewidth=0,
                    rstride=2, cstride=2, antialiased=True, zorder=1)

    fig.text(0.35, 0.955, "Metropolis–Hastings sampling", fontsize=14, ha="center")
    fig.text(0.35, 0.915, r"$\pi(x,y) \propto \exp\left(-(x^2 + y^2 + 10\sin x \cos y)\right)$",
             fontsize=10, color="#555555", ha="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=27, azim=45)

    # MH
    samples, step_records = metropolis_hastings(initial_state, iterations, proposal_std)
    z_samples = surface(samples[:, 0], samples[:, 1])

    # Animation
    animate_MH(fig, ax, samples, step_records, z_samples, panel_ax)
    plt.show()


if __name__ == "__main__":
    main()
