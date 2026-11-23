# coding: utf-8
# %%
import sys, os

# # 获取当前这个py文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # 拼接 book_src 的绝对路径
sys.path.append(os.path.join(BASE_DIR, ".."))
import numpy as np
import matplotlib.pyplot as plt

from network_2d import network_2d
from dataset_2d import get_dataset
from basic_func import plot_loss_curve

# set seed
np.random.seed(0)
# 读入数据
(x_train, t_train), (x_test, t_test) = get_dataset(
    norm=True, train_num=4000, test_num=200
)

network = network_2d(input_size=2, hidden_size=3, output_size=2)
# %% training
print("training...")
total_epoch = 30

train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size // batch_size, 1)
iters_num = iter_per_epoch * total_epoch


print(f"{'epoch/total':<15} {'tran_acc':<15} {'test_acc':<15}")
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f"{str(i//iter_per_epoch)+'/'+str(total_epoch):<15} {train_acc:<15.4f} {test_acc:<15.4f}"
        )

plot_loss_curve(train_loss_list)
trained_parm = {k: v.copy() for k, v in network.params.items()}
for key in ("W1", "b1", "W2", "b2"):
    network.params[key].round(decimals=1, out=network.params[key])

# %% test
err_idx = []
accuracy_cnt = 0
for i in range(len(x_test)):
    y = network.predict(x_test[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    t = np.argmax(t_test[i])
    if p == t:
        accuracy_cnt += 1
    else:
        err_idx += [i]

print("Test set Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
print(f"err_idx {err_idx}")
idx = 0
eid = 0
# %%
W1 = np.array([[2.1, 0.4, -2.4], [-1.7, 2.6, -1.0]])
b1 = np.array([-0.3, -0.3, -0.2])
W2 = np.array([[-1.9, 1.9], [-1.9, 1.9], [-1.8, 1.8]])
b2 = np.array([1.7, -1.7])
W1 /= 3

def forward(x):
    z1 = x @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    return z2, h


# 生成网格
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
grid = np.stack([X1, X2], axis=-1)
z, h = forward(grid)
# 看z0-z1，也就是S
S = z[..., 0] - z[..., 1]

plt.figure(figsize=(6, 6))
plt.contourf(X1, X2, S > 0, alpha=0.4, cmap="coolwarm")
plt.contour(X1, X2, S, levels=[0], colors="black")


# 画出3条ReLU分界线
def plot_line(a, b, c, label):
    # a x1 + b x2 + c =0
    xs = np.linspace(-3, 3, 100)
    ys = (-a * xs - c) / b
    mask = (ys >= -3) & (ys <= 3)
    plt.plot(xs[mask], ys[mask], "--", label=label)


def plot_circle(r=2, c="green"):
    theta = np.linspace(0, 2 * np.pi, 100)
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    plt.plot(xs, ys, color=c, linewidth=2, label=f"circle r={r}")


for i, name in enumerate(["L1", "L2", "L3"]):
    plot_line(W1[0, i], W1[1, i], b1[i], name)
plot_circle(1, "green")
plot_circle(2, "blue")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(alpha=0.2)
plt.legend()
plt.axis("equal")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision boundary (piecewise linear)")
plt.show()

# %% update parms
network.modify_parms(W1=W1 * 3, b1=b1, W2=W2, b2=b2)
