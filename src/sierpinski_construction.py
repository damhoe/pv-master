"""

Construction of Sierpinski triangle (2D fractal).

@author: Damian Hoedtke
@data: dec, 21

"""
import sys
import numpy as np

from dataclasses import dataclass

@dataclass
class Triangle:
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
        
    def area(self):
        return 0.5 * abs(self.x1 * (self.y2 - self.y3) + self.x2 * (self.y3 - self.y1) + self.x3 * (self.y1 - self.y2))
    
    def check_p(self, x, y):
        t1 = Triangle(x, y, self.x2, self.y2, self.x3, self.y3)
        t2 = Triangle(self.x1, self.y1, x, y, self.x3, self.y3)
        t3 = Triangle(self.x1, self.y1, self.x2, self.y2, x, y)
        A = self.area()
        A1 = t1.area()
        A2 = t2.area()
        A3 = t3.area()
        return abs(A - (A1 + A2 + A3)) < 1e-6
    

def sierpinski(triangle_list):
    new_list = []
    for t in triangle_list:
        dx12 = t.x1 + 0.5 * (t.x2 - t.x1)
        dy12 = t.y1 + 0.5 * (t.y2 - t.y1)
        dx23 = t.x2 + 0.5 * (t.x3 - t.x2)
        dy23 = t.y2 + 0.5 * (t.y3 - t.y2)
        dx31 = t.x3 + 0.5 * (t.x1 - t.x3)
        dy31 = t.y3 + 0.5 * (t.y1 - t.y3)
        t1 = Triangle(t.x1, t.y1, dx12, dy12, dx31, dy31)
        t2 = Triangle(t.x2, t.y2, dx12, dy12, dx23, dy23)
        t3 = Triangle(t.x3, t.y3, dx31, dy31, dx23, dy23)
        new_list.append(t1)
        new_list.append(t2)
        new_list.append(t3)
    return new_list


def main(N):
    #N = 20000
    D = 2
    data = np.random.rand(D * N).reshape(N, D) * (1.0 + 1.0) - 1.0
    selected = data

    sqrt3h = 0.5 * np.sqrt(3)
    t = Triangle(-1.0, -sqrt3h, 1.0, -sqrt3h, 0.0, sqrt3h)
    #t = Triangle(-1.0, -1.0, 1.0, -1.0, 0.0, 1.0)
    print(f'{t.area():.2f}')
    print(f'{t.check_p(1, 2)}')

    triangle_list = [t]

    n = 7
    for i in range(n):
        new = []
        for point in selected:
            for t in triangle_list:
                if t.check_p(point[0], point[1]):
                    new.append(point)
        selected = np.array(new)
        triangle_list = sierpinski(triangle_list)

    # save results
    fname = f'data/sierpinski/points_{N}.csv'
    np.savetxt(fname, selected, fmt='%.18e')


if __name__ == '__main__':
    N = int(sys.argv[1])
    main(N)
