import matplotlib.pyplot as plt


def plot_vector_field(X, Y, U, V):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.streamplot(X, Y, U, V, density=1.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase portrait")

    return fig, ax


def plot_trajectory(ax, x_traj, y_traj):
    ax.plot(x_traj, y_traj, linewidth=2)
    ax.plot(x_traj[0], y_traj[0], "o")  # početna tačka
