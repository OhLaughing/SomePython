import matplotlib.pyplot as plt
import numpy as np


def plot_an_image(image, label):
    # pick_one = random.randint(0, 5000)
    # image = x[pick_one, :]
    image = image.reshape(20, 20).transpose()
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(label))


# 数字识别，把原来的图x轴和y轴移动， x_dif\如果为正，向右移动，如果为负向左移动
# y_dif\如果为正，向上移动，如果为负向下移动
def move(data, column, row, x_dif, y_dif):
    assert len(data) == column * row

    x_start = 0 if x_dif > 0 else -x_dif
    y_start = y_dif if y_dif > 0 else 0
    dat = data.reshape(column, row)
    if x_dif > 0:
        dat = np.hstack((np.zeros((column, x_dif)), dat))
    else:
        dat = np.hstack((dat, np.zeros((column, -x_dif))))

    if y_dif > 0:
        dat = np.vstack((dat, np.zeros((y_dif, row + np.abs(x_dif)))))
    else:
        dat = np.vstack((np.zeros((-y_dif, row + np.abs(x_dif))), dat))

    # 行对应y轴，列对应x轴，不要搞混
    dat = dat[y_start:y_start + column, x_start:x_start + row]
    print(dat.shape)
    return np.ravel(dat)

a = np.arange(12)
print(a.reshape(3,4))

b = move(a,3,4,-1,-1)
print(b.reshape(3,4))
