# coding: utf-8
# %%
import sys, os

# 获取当前这个py文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 拼接 book_src 的绝对路径
sys.path.append(os.path.join(BASE_DIR, "..", "book_src"))
import numpy as np
from orange_fcn import orange_fcn
from dataset.mnist import load_mnist
from basic_func import draw_pic

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = orange_fcn(input_size=784, hidden_size=50, output_size=10)


def analyse_pic(i, x=x_test, t=t_test):
    a, b = np.argmax(network.predict(x[i])), np.argmax(t[i])

    draw_pic(
        x[i].reshape(28, 28),
        f"{'good' if a==b else 'bad'} {a} vs {b} idx: {i}",
    )


# %% training
print("training...")
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

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
            f"epoch {i/iter_per_epoch}, tran_acc {train_acc:.4f}, test_acc {test_acc:.4f}"
        )
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
analyse_pic(idx, x_test, t_test)
idx += 1
# %% show error
i = err_idx[eid]
analyse_pic(i, x_test, t_test)
eid += 1
