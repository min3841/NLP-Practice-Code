def AND_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

def NAND_gate(x1, x2):
    w1 = -0.5
    w2 = -0.5
    b = 0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


def OR_gate(x1, x2):
    w1 = 0.6
    w2 = 0.6
    b = -0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1



def XOR_gate(x1,x2):
    X = NAND_gate(x1,x2)
    Y = OR_gate(x1,x2)
    return AND_gate(X,Y)


print(XOR_gate(0,0))
print(XOR_gate(0,1))
print(XOR_gate(1,0))
print(XOR_gate(1,1))

