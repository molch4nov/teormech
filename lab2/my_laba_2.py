import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    t = np.linspace(0, 20, 1000)

    x = abs(np.cos(1.9 * t)) + np.sin(2 * t) + 2
    s = abs(x)
    L = 4.5
    l = 3
    A = 2
    w = 2
    phi_o = 22.5
    phi = A * np.sin(w * t + phi_o)
    # точечка О
    X_o = 6
    Y_o = 5
    #основание над точечкой О
    Bx = X_o
    By = Y_o
    Cx = Bx - 0.2;
    Cy = By + 0.5
    Dx = Bx + 0.2;
    Dy = Cy
    # штучка для стержня
    X_l = X_o + L * np.cos(phi)
    Y_l = Y_o + L * np.sin(phi)
    # пружинка
    #матрица поворота
    def Rot2D(X, Y, Phi):
        RotX = X * np.cos(Phi) - Y * np.sin(Phi)
        RotY = X * np.sin(Phi) + Y * np.cos(Phi)
        return RotX, RotY
    n = 20
    sh = 0.25
    b = 1 / (n - 2)
    x_P = np.zeros(n)
    y_P = np.zeros(n)
    x_P[0] = 0
    x_P[n - 1] = 1
    y_P[0] = 0
    y_P[n - 1] = 0
    for i in range(n - 2):
        x_P[i + 1] = b * (i + 1) - b / 2
        y_P[i + 1] = sh * (-1) ** i
    #колечко
    X_m = X_o + s * np.cos(phi)
    Y_m = Y_o + s * np.sin(phi)

    Vx0 = np.diff(X_m )
    Vy0 = np.diff(Y_m )

    #отрисовочка
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(0, 13)
    plt.ylim(0, 13)
    ax.set_aspect(1)

    #наши линии и точечки на графике
    LineBC = ax.plot([Bx, Cx], [By, Cy], c='k')[0]
    LineCD = ax.plot([Cx, Dx], [Cy, Dy], c='k')[0]
    LineDB = ax.plot([Dx, Bx], [Dy, By], c='k')[0]
    Line_OM = ax.plot([X_o,X_l[0]],[Y_o, Y_l[0]], lw = 5)[0]
    O, = ax.plot(X_o, Y_o, marker='o', ms=10, mec='k', c='w')
    Rx, Ry = Rot2D(x_P * s[0], y_P, phi[0])
    Spring = ax.plot(Rx + X_o, Ry + Y_o, c = 'm')[0]
    Point_M = ax.plot(X_m[0], Y_m[0], marker='o', ms=15, mew=3, mec="orange", c='w')[0]

    ax2 = fig.add_subplot(4, 2, 1)
    ax2.plot(Vx0)
    plt.title('Vx of circle')
    plt.xlabel('t values')
    plt.ylabel('Vx values')

    ax3 = fig.add_subplot(4, 2, 2)
    ax3.plot(Vy0)
    plt.title('Vy of circle')
    plt.xlabel('t values')
    plt.ylabel('Vy values')

    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    #ну собственно кино
    def Kino(i):
        LineBC.set_data([Bx, Cx], [By, Cy])
        LineCD.set_data([Cx, Dx], [Cy, Dy])
        LineDB.set_data([Dx, Bx], [Dy, By])
        Rx, Ry = Rot2D(x_P * s[i],y_P, phi[i])
        Spring.set_data(Rx + X_o, Ry + Y_o)
        Point_M.set_data(X_m[i] , Y_m[i])
        Line_OM.set_data([X_o,X_l[i]],[Y_o, Y_l[i]])
        return [LineBC, LineDB, LineCD, Spring, Line_OM,Point_M]

    anima = FuncAnimation(fig, Kino,  frames=1000, interval=100)

    plt.show()
