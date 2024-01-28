#The concept belongs to Andrew Polar and Mike Poluektov.
#details:
#https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742?via%3Dihub
#https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149?via%3Dihub
#readers friendly format ezcodesample.com

import numpy as np
import math

#next three classes make Kolmogorov-Arnold representation
class PLL:
    def __init__(instance, xmin, xmax, points):
        instance.Initialize(xmin, xmax, points)

    def Initialize(instance, xmin, xmax, points):
        if (points < 2):
            print("Fatal: number of blocks is too low")
            exit(0)
        if (xmin >= xmax):
            xmax = xmin + 0.5
            xmin -= 0.5
               
        instance._xmin = xmin
        instance._xmax = xmax
        instance._points = points

        instance._deltax = (instance._xmax - instance._xmin) / (instance._points - 1)
        instance._y = []
        for idx in range(instance._points): 
            instance._y.append(0.0)

    def SetRandom(instance, ymin, ymax):
        for idx in range(instance._points):
            instance._y[idx] = np.random.randint(10, 1000)
        
        min = np.min(instance._y)
        max = np.max(instance._y)
        if (min == max):
            max = min + 1.0

        for idx in range(instance._points):
            instance._y[idx] = (instance._y[idx] - min) / (max - min)
            instance._y[idx] = instance._y[idx] * (ymax - ymin) + ymin

    def GetDerrivative(instance, x):
        length = len(instance._y)
        low = int((x - instance._xmin) / instance._deltax)
        if (low < 0): low = 0
        if (low > length - 2): low = length - 2
        return (instance._y[low + 1] - instance._y[low]) / instance._deltax
    
    def Update(instance, x, delta, mu):
        length = len(instance._y)
        if (x < instance._xmin):
            instance._deltax = (instance._xmax - x) / (instance._points - 1)
            instance._xmin = x
 
        if (x > instance._xmax):
            instance._deltax = (x - instance._xmin) / (instance._points - 1)
            instance._xmax = instance._xmin + (instance._points - 1) * instance._deltax

        left = int((x - instance._xmin) / instance._deltax)
        if (left < 0): left = 0
        if (left >= length - 1):
            instance._y[length - 1] += delta * mu 
            return

        leftx = x - (instance._xmin + left * instance._deltax)
        rightx = instance._xmin + (left + 1) * instance._deltax - x
        instance._y[left + 1] += delta * leftx / instance._deltax * mu
        instance._y[left] += delta * rightx / instance._deltax * mu

    def GetFunctionValue(instance, x):
        length = len(instance._y)
        if (x < instance._xmin):
            derrivative = (instance._y[1] - instance._y[0]) / instance._deltax
            return instance._y[1] - derrivative * (instance._xmin + instance._deltax - x)

        if (x > instance._xmax):
            derrivative = (instance._y[length - 1] - instance._y[length - 2]) / instance._deltax
            return instance._y[length - 2] + derrivative * (x - (instance._xmax - instance._deltax))

        left = int((x - instance._xmin) / instance._deltax)
        if (left < 0): left = 0
        if (left >= length - 1):
            return instance._y[length - 1]
        
        leftx = x - (instance._xmin + left * instance._deltax)
        return (instance._y[left + 1] - instance._y[left]) / instance._deltax * leftx + instance._y[left]        

class U:
    def __init__(instance, xmin, xmax, targetMin, targetMax, layers):
        length = len(layers)
        ymin = targetMin / length
        ymax = targetMax / length
        instance._plist = []
        for idx in range(0, length):
            pll = PLL(xmin[idx], xmax[idx], layers[idx])
            instance._plist.append(pll)
        instance.SetRandom(ymin, ymax)

    def Clear(instance):
        instance._plist.clear()

    def GetDerrivative(instance, layer, x):
        return instance._plist[layer].GetDerrivative(x)
    
    def SetRandom(instance, ymin, ymax):
        length = len(instance._plist)
        for pll in instance._plist: 
            pll.SetRandom(ymin / length, ymax / length)

    def Update(instance, delta, inputs, mu):
        i = 0
        length = len(inputs)
        for i in range(0, length):
            instance._plist[i].Update(inputs[i], delta / length, mu)
 
    def GetU(instance, inputs):
        f = 0.0
        length = len(inputs)
        for i in range(0, length):
            f += instance._plist[i].GetFunctionValue(inputs[i])
        return f
    
class KolmogorovModel:
    #model configuration
    points_in_interior = 5
    points_in_exterior = 15
    muRoot = 0.2
    muLeaves = 0.05
    nEpochs = 32
    nLeaves = -1 #negative number means it is chosen according to theory

    def __init__(instance, inputs, target):
        instance._inputs = inputs
        instance._target = target
        lenInput = len(instance._inputs)
        lenTarget = len(instance._target)
        if (lenInput != lenTarget):
            print("Fatal: the data is misformatted")
            exit(0)
        instance.FindMinMax()

        number_of_inputs = len(instance._inputs[0])
        if (instance.nLeaves < 0):
            instance.nLeaves = number_of_inputs * 2 + 1

        instance._interior_structure = []
        instance._exterior_structure = []

        for idx in range(0, number_of_inputs):
            instance._interior_structure.append(instance.points_in_interior)

        for idx in range(0, instance.nLeaves):
            instance._exterior_structure.append(instance.points_in_exterior)

        instance.GenerateInitialOperators()

    def FindMinMax(instance):
        instance._targetMin = np.min(instance._target)
        instance._targetMax = np.max(instance._target)

        cols = len(instance._inputs[0])
        rows = len(instance._inputs)

        instance._xmin = []
        instance._xmax = []
        for j in range(0, cols):
            xmin = instance._inputs[0][j]
            xmax = instance._inputs[0][j]
            for i in range(0, rows):
                if (instance._inputs[i][j] < xmin): xmin = instance._inputs[i][j]
                if (instance._inputs[i][j] > xmax): xmax = instance._inputs[i][j]
            instance._xmin.append(xmin)
            instance._xmax.append(xmax)
                
    def GenerateInitialOperators(instance): 
        instance._ulist = []
        instance._ulist.clear()
 
        for counter in range(0, instance.nLeaves):
            uc = U(instance._xmin, instance._xmax, instance._targetMin, instance._targetMax, instance._interior_structure)
            instance._ulist.append(uc)
        
        mmin = []
        mmax = []

        for idx in range(0, instance.nLeaves):
            mmin.append(instance._targetMin)
            mmax.append(instance._targetMax)

        instance._bigU = U(mmin, mmax, instance._targetMin, instance._targetMax, instance._exterior_structure) 
        
    def GetVector(instance, data):
        size = len(instance._ulist)
        vector = []
        for idx in range(0, size):
            vector.append(instance._ulist[idx].GetU(data))
        return vector

    def ComputeOutput(instance, inputs):
        v = instance.GetVector(inputs)
        output = instance._bigU.GetU(v)
        return output

    def BuildRepresentation(instance):
        size = len(instance._inputs)
        ulistSize = len(instance._ulist)
        for step in range(0, instance.nEpochs):
            dist = 0.0
            for i in range(0, size):
                v = instance.GetVector(instance._inputs[i])
                model = instance._bigU.GetU(v)
                diff = instance._target[i] - model
                relative_diff = diff / len(v)
                for k in range(0, ulistSize):
                    if (v[k] > instance._targetMin and v[k] < instance._targetMax):
                        derrivative = instance._bigU.GetDerrivative(k, v[k])
                        instance._ulist[k].Update(relative_diff * derrivative, instance._inputs[i], instance.muLeaves)
                        
                instance._bigU.Update(diff, v, instance.muRoot)
                dist += diff * diff
            
            dist /= size
            dist = np.math.sqrt(dist)
            dist /= (instance._targetMax - instance._targetMin)
            print(f"Step {step}, relative error {dist}")
#end of description part, now we test it
        
#Now we generate training and validation sample
def compute_target(z0, z1, z2, z3, z4):
    pi = 3.14159265359
    p = 1.0 / pi
    p *= 2.0 + 2.0 * z2
    p *= 1.0 / 3.0
    p *= math.atan(20.0 * math.exp(z4) * (z0 - 0.5 + z1 / 6.0)) + pi / 2.0
    q = 1.0 / pi
    q *= 2.0 + 2.0 * z3
    q *= 1.0 / 3.0
    q *= math.atan(20.0 * math.exp(z4) * (z0 - 0.5 - z1 / 6.0)) + pi / 2.0
    return p + q 

inputs = []
target = []
rows, cols = 10000, 5
for i in range(rows):
    col = []
    for j in range(cols):
        col.append(np.random.randint(0, 1000) / 1000.0)
    inputs.append(col)
    target.append(compute_target(col[0], col[1], col[2], col[3], col[4]))
#training dataset is generated

km = KolmogorovModel(inputs, target)
km.BuildRepresentation()
