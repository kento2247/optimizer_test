def least_squares():
    X:list = [1, 2, 3, 4, 5]  # 説明変数
    Y:list = [2, 4, 5, 4, 5]  # 目的変数
    mean_x:float = sum(X) / len(X) # 平均値を計算
    mean_y:float = sum(Y) / len(Y) # 平均値を計算
    covariance:float = sum((x - mean_x) * (yi - mean_y) for x, yi in zip(X, Y)) / len(X) # 共分散を計算
    variance:float = sum((x - mean_x) ** 2 for x in X) / len(X) # 分散を計算

    slope:float = covariance / variance # 傾きaを計算
    intercept:float = mean_y - slope * mean_x # 切片bを計算

    print("説明変数が一つと仮定した場合(単回帰):\n y =", slope, "* x +", intercept) # y = 0.6 * x + 2.2
    plot(X,Y,slope,intercept) # グラフを描画

def np_least_squares():
    import numpy as np
    X:np.ndarray = np.array([1, 2, 3, 4, 5])  # 説明変数
    Y:np.ndarray = np.array([2, 4, 5, 4, 5])  # 目的変数

    slope:float = np.polyfit(X, Y, 1)[0] # 傾きaを計算
    intercept:float = np.polyfit(X, Y, 1)[1] # 切片bを計算

    print("説明変数が一つと仮定した場合(単回帰):\n y =", slope, "* x +", intercept) # y = 0.6 * x + 2.2
    plot(X,Y,slope,intercept) # グラフを描画

def gradient_descent():
    import numpy as np
    X:np.ndarray = np.array([1, 2, 3, 4, 5])  # 説明変数
    Y:np.ndarray = np.array([2, 4, 5, 4, 5])  # 目的変数
    learning_rate:float = 0.01 # 学習率
    epochs:int = 3000 # エポック数
    slope:float = 0 # 傾きaの初期値
    intercept:float = 0 # 切片bの初期値

    for epoch in range(epochs):
        y_pred:np.ndarray = slope * X + intercept # 予測値を計算
        error:np.ndarray = Y - y_pred # 誤差を計算
        slope += learning_rate * np.sum(error * X) / len(X) # 傾きaを更新
        intercept += learning_rate * np.sum(error) / len(X) # 切片bを更新

    print("説明変数が一つと仮定した場合(単回帰):\n y =", slope, "* x +", intercept) # y = 0.6 * x + 2.2
    plot(X,Y,slope,intercept) # グラフを描画

def plot(X,Y,slope,intercept):
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array(X)
    y = np.array(Y)
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, color='red')
    plt.show()

if __name__ == '__main__':
    # least_squares() # 最小二乗法
    # np_least_squares() # numpyの最小二乗法
    gradient_descent() # 勾配降下法