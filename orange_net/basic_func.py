# %%
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


def softmax(x: NDArray):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y: NDArray, t: NDArray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class relu:
    def __init__(self):
        self.mask = None

    def forward(self, input: NDArray):
        self.mask = input <= 0
        output = input
        output[self.mask] = 0

        return output

    def backward(self, dout: NDArray):
        dout[self.mask] = 0
        dinput = dout

        return dinput


class softmax_with_loss:
    def __init__(self):
        self.out = np.empty(0)

    def forward(self, input: NDArray, label: NDArray):
        self.input = input
        self.out = softmax(input)
        self.label = label
        # cross_entropy_error函数中会把label修改为非 one-hot 编码
        self.loss = cross_entropy_error(self.out, label.copy())
        return self.loss

    def backward(self):
        batch_size = self.input.shape[0]
        if self.label.size == self.out.size:  # 监督数据是one-hot-vector的情况
            dx = self.out - self.label
        else:
            dx = self.out.copy()
            dx[np.arange(batch_size), self.label] -= 1
        return dx / batch_size


class affine:
    def __init__(self, w: NDArray, b: NDArray):
        self.w, self.b = w, b

    def forward(self, input: NDArray):
        if input.ndim == 1:
            # if don't have batch dim
            input = input.reshape(1, -1)
        else:
            # have batch dim
            input = input.reshape(input.shape[0], -1)
        self.input = input
        return input @ self.w + self.b

    def backward(self, dout: NDArray):
        self.dw = self.input.T @ dout
        # 后层在最初计算反向传播的时候，已经除以 batchsize 了，所以这里直接加就可以了
        self.db = dout.sum(axis=0)
        dx = dout @ self.w.T
        return dx


def draw_pic(pic: NDArray, title: str):
    vmin, vmax = pic.min(), pic.max()
    plt.imshow(pic, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_loss_curve(loss_list: list[float], title: str = "Loss Curve") -> None:
    """
    绘制训练损失曲线
    :param loss_list: 每一轮/每一批收集的loss
    :param title: 图标题
    """
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, color="#2E86AB", linewidth=1.5, label="train loss")
    plt.xlabel("iteration / epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    draw_pic(img, "test")
    loss_list = [0.1 * i for i in range(10)]
    loss_list.reverse()
    plot_loss_curve(loss_list)
