import numpy as np

x_lower_bound = -10
x_upper_bound = 10
y_lower_bound = -10
y_upper_bound = 10
x, y = np.array(np.meshgrid(np.linspace(x_lower_bound, x_upper_bound,100), np.linspace(y_lower_bound,y_upper_bound,100)))

def camel(x,y):
    return  2.0*x**2 - 1.05*x**4 + (x**6 / 6.0) + x*y + y**2

def matyas(x,y):
    return 0.26*(x**2 + y**2) - 0.48*x*y

def schaffer(x,y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5)/(1+0.001*(x**2+y**2))**2