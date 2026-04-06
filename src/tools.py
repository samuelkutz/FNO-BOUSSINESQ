import torch
from timeit import default_timer
from torch.optim import Adam

from FNO.fno_model import FNO2d, L2_loss


# Train FNO model
def train_model(x_train, y_train, epochs=3000, batch_size=16, lr=1e-3,
                modes1=16, modes2=16, width=32, device='cpu'):
    """Train FNO model. Returns (model, train_loss_history)"""
    model = FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = L2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss_history = []

    print("Starting training...")
    t1 = default_timer()
    for ep in range(epochs):
        model.train()
        train_l2 = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        train_l2 /= len(train_loader)
        train_loss_history.append(train_l2)

        if ep % 100 == 0:
            print(f"Epoch: {ep}, Time: {t1:.2f}s, Loss: {train_l2}")

    return model, train_loss_history

# Save model state dict
def save_model(model, filepath=None, *, epochs=None, n_samples=None, modes=(16,16), width=None, extra=''):
    """Save model into RESULTS with informative filename when filepath is None.
    If filepath provided, use it directly.
    """
    import os
    if filepath is None:
        os.makedirs('RESULTS', exist_ok=True)
        modes1, modes2 = modes
        parts = [f'fno_epochs{epochs}', f'samples{n_samples}', f'modes{modes1}x{modes2}']
        if width is not None:
            parts.append(f'width{width}')
        if extra:
            parts.append(extra)
        filename = '_'.join([p for p in parts if p]) + '.pth'
        filepath = os.path.join('RESULTS', filename)

    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


# Load model state dict
def load_model(filepath, device='cpu'):
    model = FNO2d(modes1=16, modes2=16, width=32).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model


# Multi-resolution test (returns relative errors per resolution and predictions)
def test_multi_resolution(model, test_resolutions, val_a=1.0, val_b=1.0, device='cpu'):
    from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
    import numpy as np

    results = {}
    model.eval()
    for res in test_resolutions:
        bsq_test = Boussinesq(-30, 30, 0, 15, val_a, val_b)
        solver_test = PseudoSpectralBoussinesq(bsq_test, Nx=res, Nt=res-1, device=device)
        x_test, t_test, eta_true, u_true = solver_test.solve()

        ch0 = np.tile(eta_true[0:1, :].T, (1, res))
        ch1 = np.tile(u_true[0:1, :].T, (1, res))
        ch2 = np.ones((res, res)) * val_a
        ch3 = np.ones((res, res)) * val_b

        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_tensor = model(input_tensor)
            eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]

        eta_true_t = eta_true.T
        diff = np.abs(eta_true_t - eta_pred)
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(eta_true_t) + 1e-8)

        results[res] = {
            'x': x_test,
            't': t_test,
            'eta_true': eta_true_t,
            'eta_pred': eta_pred,
            'rel_error': float(rel_error)
        }
    return results
