# %%
import numpy as np
import matplotlib.pyplot as plt


def sample_disk(n: int, r_inner: float, r_outer: float):
    """
    采样圆环 r_inner <= r < r_outer 内均匀随机点
    n: 点数
    返回 (x,y) shape [n,2]
    """
    # 面积均匀：r^2 均匀分布
    u = np.random.rand(n)
    r_sq = r_inner**2 + u * (r_outer**2 - r_inner**2)
    r = np.sqrt(r_sq)
    theta = np.random.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def _get_dataset(num=400):
    # tran data set
    samples = num // 2
    inner = sample_disk(samples, 0, 1)
    inner_t = np.array([[1, 0]] * samples)
    outer = sample_disk(samples, 2, 3)
    outer_t = np.array([[0, 1]] * samples)
    x = np.concatenate([inner, outer], axis=0)
    t = np.concatenate([inner_t, outer_t], axis=0)
    return x, t


def get_dataset(norm=True, train_num=400, test_num=200):
    x_train, t_train = _get_dataset(train_num)
    x_test, t_test = _get_dataset(test_num)
    return (x_train / 3, t_train), (x_test / 3, t_test)


def draw_circle(pts_inner, pts_ring, title=""):
    # # 1. 单位圆内：r < 1
    # pts_inner = sample_disk(200, 0.0, 1.0)
    # # 2. 圆环：2 < r < 3
    # pts_ring = sample_disk(300, 2.0, 3.0)

    plt.scatter(pts_inner[:, 0], pts_inner[:, 1], s=8)
    plt.scatter(pts_ring[:, 0], pts_ring[:, 1], s=8)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.show()


def draw_from_dataset(x, t, title=""):
    inner_mask = t[:, 0] == 1
    outer_mask = t[:, 1] == 1
    draw_circle(x[inner_mask], x[outer_mask], title)


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = get_dataset()
    draw_from_dataset(x_train, t_train, "train")
    draw_from_dataset(x_test, t_test, "test")
