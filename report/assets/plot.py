#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import json


def plot_accuracy(ax, log):
    ax.set_title('Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(list(range(0, len(log) + 1, 10)))
    ax.set_ylim(0.6, 1)

    def plot(is_train):
        if is_train:
            key = 'main/accuracy'
            label = 'Training'
        else:
            key = 'validation/main/accuracy'
            label = 'Validation'

        xs = [e['epoch'] for e in log]
        ys = [e[key] for e in log]
        ax.plot(xs, ys, label=label)

    plot(True)
    plot(False)
    ax.legend(loc='upper left')


def main():
    with open('log') as f:
        log = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_accuracy(ax, log)
    fig.tight_layout()
    fig.savefig('accuracy.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
