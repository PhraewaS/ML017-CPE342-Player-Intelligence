"""Task 2 building blocks: Gaussian-smoothed class-conditional histograms (feature selection by mutual
information) and a decision rule that maximises macro F1 from posterior probabilities."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def fit_density(histogram, features, labels):
    model = histogram(80, 1, skip=('random_metric_1', 'random_metric_2', 'random_metric_3'))
    model.fit(features, labels)
    prior = np.exp(model.prior)
    selected = []
    for name in model.cols:
        frequencies = np.exp(model.logp[name])
        if name in model.edges:
            frequencies[:, :-1] = gaussian_filter1d(frequencies[:, :-1], 1.25, axis=1, mode='reflect')
            frequencies /= frequencies.sum(axis=1, keepdims=True)
        pooled = prior @ frequencies
        info = (prior[:, None] * frequencies * np.log(frequencies / pooled)).sum()
        if info > 0.0015:
            selected.append(name)
            model.logp[name] = np.log(frequencies)
    model.cols = selected
    return model


def macro_f1_rule(posteriors):
    true_rate = posteriors.mean(axis=0)
    predicted = posteriors.argmax(axis=1)
    for _ in range(12):
        assigned = np.eye(4)[predicted]
        total_rate = true_rate + assigned.mean(axis=0)
        offset = (assigned * posteriors).mean(axis=0) / total_rate
        predicted = ((posteriors - offset) / total_rate).argmax(axis=1)
    return offset, total_rate
