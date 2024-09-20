from math import pi, sqrt

n = 20
lmbd = 0.8

# Exact QFIs for model 1 #
# calculate exact QFI
QFI_exact = (1-(sqrt(1-lmbd))**n)**2/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd)**(n-1)\
            + (n-1)*lmbd/(1-sqrt(1-lmbd))**2 \
            + (1-lmbd - (1-lmbd)**n)/(1-sqrt(1-lmbd))**2 \
            - 2*lmbd*(sqrt(1-lmbd)-(sqrt(1-lmbd))**n)/((1-sqrt(1-lmbd))**3) \
            + (1- (1-lmbd)**(n-1))/lmbd \
            -(1-lmbd-(1-lmbd)**n)/lmbd - 1 + n*(1-lmbd)**(n-1)\
            +n*(1- (1-lmbd)**(n-1))
print('Exact QFI is {}'.format(4*QFI_exact))