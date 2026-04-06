import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq


# Plot training loss
def plot_training_loss(train_loss_history, outdir='RESULTS', filename=None):
    import os
    os.makedirs(outdir, exist_ok=True)
    if filename is None:
        filename = 'training_loss.png'
    outpath = os.path.join(outdir, filename)

    plt.figure()
    plt.plot(train_loss_history)
    plt.title("Training Loss History")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(outpath, dpi=150)
    print(f"Training loss plot saved to {outpath}")
    plt.close()


# Create animation comparing FNO vs ground-truth
def generate_animation(model, val_a=1, val_b=1, res_high=64, device='cpu', outdir='RESULTS'):
    import os
    os.makedirs(outdir, exist_ok=True)

    bsq_high = Boussinesq(x_min=-30, x_max=30, t_min=0, t_max=15, a=val_a, b=val_b)
    solver_high = PseudoSpectralBoussinesq(bsq_high, Nx=res_high, Nt=res_high-1, device=device)
    x_high, t_high, eta_true, u_true = solver_high.solve()

    ch0 = np.tile(eta_true[0:1, :].T, (1, res_high))
    ch1 = np.tile(u_true[0:1, :].T, (1, res_high))
    ch2 = np.ones((res_high, res_high)) * val_a
    ch3 = np.ones((res_high, res_high)) * val_b

    input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
    input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(input_tensor)
        eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]
        u_pred = pred_tensor.squeeze().cpu().numpy()[1, :, :]

    eta_true = eta_true.T
    u_true = u_true.T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    fig.suptitle(f'Boussinesq FNO (a={val_a}, b={val_b})', fontsize=16)

    ax1.set_ylabel(r'$\eta(x, t)$', fontsize=12)
    ax1.set_xlim(x_high[0], x_high[-1])
    ax1.set_ylim(np.min(eta_true) - 0.2, np.max(eta_true) + 0.2)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xticklabels([])

    l1_pred, = ax1.plot([], [], lw=2.5, color='#ff7f0e', label='FNO')
    l1_true, = ax1.plot([], [], lw=2, color='black', linestyle='--', alpha=0.5, label='Pseudospectral')
    ax1.legend(loc='upper right')

    ax2.set_ylabel(r'$u(x, t)$', fontsize=12)
    ax2.set_xlabel('x (Space)', fontsize=12)
    ax2.set_xlim(x_high[0], x_high[-1])
    ax2.set_ylim(np.min(u_true) - 0.2, np.max(u_true) + 0.2)
    ax2.grid(True, linestyle=':', alpha=0.6)

    l2_pred, = ax2.plot([], [], lw=2.5, color='#ff7f0e', label='FNO')
    l2_true, = ax2.plot([], [], lw=2, color='black', linestyle='--', alpha=0.5, label='Pseudospectral')

    time_text = ax1.text(0.02, 0.9, '', transform=ax1.transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def init():
        l1_pred.set_data([], [])
        l1_true.set_data([], [])
        l2_pred.set_data([], [])
        l2_true.set_data([], [])
        time_text.set_text('')
        return l1_pred, l1_true, l2_pred, l2_true, time_text

    def update(frame):
        l1_pred.set_data(x_high, eta_pred[:, frame])
        l1_true.set_data(x_high, eta_true[:, frame])
        l2_pred.set_data(x_high, u_pred[:, frame])
        l2_true.set_data(x_high, u_true[:, frame])
        time_text.set_text(f'Time: {t_high[frame]:.2f}')
        return l1_pred, l1_true, l2_pred, l2_true, time_text

    ani = animation.FuncAnimation(fig, update, frames=res_high,
                                  init_func=init, blit=True, interval=50)

    filename = f'boussinesq_a={val_a:.2f}_b={val_b:.2f}_res{res_high}.gif'
    outpath = os.path.join(outdir, filename)
    ani.save(outpath, writer='pillow', fps=24)
    print(f"Animation saved to {outpath}")
    plt.close(fig)


# Evaluate errors for test parameters
def evaluate_errors(model, test_params, res_high=64, device='cpu', outdir='RESULTS', filename=None):
    import os
    os.makedirs(outdir, exist_ok=True)
    if filename is None:
        filename = 'error_analysis.png'
    outpath = os.path.join(outdir, filename)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(r'Error Analysis FNO: Boussinesq ($\eta$)', fontsize=18, y=0.95)

    print(f"Evaluating {len(test_params)} cases at resolution {res_high}x{res_high}...")

    for idx, val in enumerate(test_params):
        bsq_sim = Boussinesq(x_min=-30, x_max=30, t_min=0, t_max=15, a=val, b=val)
        solver = PseudoSpectralBoussinesq(bsq_sim, Nx=res_high, Nt=res_high-1, device=device)
        x_sol, t_sol, eta_true, u_true = solver.solve()

        eta_0 = eta_true[0, :]
        u_0 = u_true[0, :]

        ch0 = np.tile(eta_0[:, None], (1, res_high))
        ch1 = np.tile(u_0[:, None], (1, res_high))
        ch2 = np.ones((res_high, res_high), dtype=np.float32) * val
        ch3 = np.ones((res_high, res_high), dtype=np.float32) * val

        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]

        eta_true_t = eta_true.T

        diff = eta_true_t - eta_pred
        norm_diff = np.linalg.norm(diff, ord=2, axis=0)
        norm_true = np.linalg.norm(eta_true_t, ord=2, axis=0)
        relative_error_t = norm_diff / (norm_true + 1e-8)

        ax_map = axes[idx, 0]
        im = ax_map.imshow(np.abs(diff).T,
                           extent=[x_sol[0], x_sol[-1], t_sol[0], t_sol[-1]],
                           origin='lower', aspect='auto', cmap='inferno')
        ax_map.set_title(rf'Absolute Error $|\eta - \tilde{{\eta}}|$ ($\alpha=\beta={val}$)')
        ax_map.set_ylabel('Time (t)')
        if idx == 2:
            ax_map.set_xlabel('Space (x)')

        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax_err = axes[idx, 1]
        ax_err.plot(t_sol, relative_error_t, color='crimson', lw=2)
        ax_err.set_title(r'Relative Error: $\frac{||\eta(\cdot,t) - \tilde{\eta}(\cdot,t)||}{||\eta(\cdot,t)||}$')
        ax_err.grid(True, linestyle='--', alpha=0.6)
        ax_err.set_xlim(t_sol[0], t_sol[-1])
        ax_err.set_ylim(bottom=0)

        mean_err = np.mean(relative_error_t)
        ax_err.text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=ax_err.transAxes,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

        if idx == 2:
            ax_err.set_xlabel('Time (t)')

        print(f"Case alpha={val}: Mean Relative Error = {mean_err:.4e}")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Error analysis plot saved to {outpath}")
    plt.close()
