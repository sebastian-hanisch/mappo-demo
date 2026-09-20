"""Kleine numpy-Bausteine für die PPO-Netze: ein Batch-MLP mit manuellem Backprop, ein
Adam-Optimierer mit Gradienten-Norm-Clipping und Softmax. Kein torch - das Portfolio bleibt
numpy-only und damit auf Streamlit Cloud lauffähig.

Ein `MLP` hält G unabhängige Netze gleicher Form und wertet sie per Batch-Matmul aus
(X hat die Form [G, N, Eingang]). Geteilte Parameter (Yu et al.) sind der Fall G=1: alle Agenten
laufen durch dasselbe Netz."""

import numpy as np


def softmax(logits):
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class MLP:
    """G unabhängige tanh-MLPs. `last_zero=True` initialisiert die letzte Schicht mit Nullen: das Netz
    gibt dann exakt 0 aus (Softmax-Logits 0 => gierige Wahl per Tie-Break Aktion 0 => Contract Net)."""

    def __init__(self, groups, sizes, rng, last_zero=False):
        self.W = []
        self.b = []
        last = len(sizes) - 1
        for i in range(last):
            w = rng.standard_normal((groups, sizes[i], sizes[i + 1])) * (1.0 / np.sqrt(sizes[i]))
            if i == last - 1:
                w = np.zeros_like(w) if last_zero else w * 0.1
            self.W.append(w)
            self.b.append(np.zeros((groups, 1, sizes[i + 1])))

    def params(self):
        return self.W + self.b

    def forward(self, x):
        """Gibt alle Schicht-Aktivierungen zurück (hs[-1] = Ausgabe)."""
        return forward_params(self.W, self.b, x)

    def backward(self, hs, d_out):
        """Gradienten (Gewichte + Biases, in der Reihenfolge von `params()`) für d_out = dL/d_Ausgabe."""
        layers = len(self.W)
        grad_w, grad_b = [None] * layers, [None] * layers
        d = d_out
        for i in range(layers - 1, -1, -1):
            grad_w[i] = np.matmul(hs[i].transpose(0, 2, 1), d)
            grad_b[i] = d.sum(1, keepdims=True)
            if i > 0:
                d = np.matmul(d, self.W[i].transpose(0, 2, 1)) * (1 - hs[i] ** 2)
        return grad_w + grad_b


def forward_params(weights, biases, x):
    """Vorwärtslauf mit expliziten Parametern (für Snapshots und gespeicherte Ergebnisse)."""
    hs = [x]
    layers = len(weights)
    for i in range(layers):
        z = np.matmul(hs[-1], weights[i]) + biases[i]
        hs.append(np.tanh(z) if i < layers - 1 else z)
    return hs


def unpack(params):
    """params() = Gewichte + Biases => (Gewichte, Biases)."""
    half = len(params) // 2
    return params[:half], params[half:]


class Adam:
    """Adam über eine Liste von Parameter-Arrays (in place), optional mit Gradienten-Norm-Clipping."""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self, grads, max_norm=0.5):
        if max_norm:
            norm = np.sqrt(sum((g * g).sum() for g in grads))
            if norm > max_norm:
                grads = [g * (max_norm / (norm + 1e-12)) for g in grads]
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
