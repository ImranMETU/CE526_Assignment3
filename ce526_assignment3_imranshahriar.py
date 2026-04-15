# -*- coding: utf-8 -*-
"""CE526_assignment3_imranshahriar.ipynb


This script has been created by Imran Shahriar to provide easy reproduction of the results provided in the third assignment of CE526-Finite Element Method instructed by Prof. Aysegul Askan Gundogan in 25/26 Spring.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad

"""Input parameters"""

L = 1.0   # Bar length, in m
f = 1000  # Body force, in N/m
T = 10000 # Traction force, in N

DBCStart = 1 # Existance of Dirichlet BC at x=0
DBCEnd   = 0 # Existance of Dirichlet BC at x=L
u0 = 0       # Dirichlet BC at x=0
uL = None    # Dirichlet BC at x=L
dx = 0.001   # Length increment, in m

"""Construction of the global stiffness matrix"""

def GlobalStiff(h, L):
    x = 0
    nEl = int(L/h) # Number of elements
    nDOF = nEl+1   # Number of DOFs
    Kglobal = np.zeros((nDOF, nDOF)) # Global stiffness matrix
    elCon = np.zeros((nEl,2))        # Element connectivity matrix

    for i in range(0,len(elCon)):
        for j in range(0,len(elCon.T)):
            elCon[i,j] = i+j

    def ElStiff(x, h):

        def Integrand_kii(ksi, h):
            return (10**5)*(1+2*ksi)*((1/h)**2)

        def Integrand_kij(ksi, h):
            return (10**5)*(1+2*ksi)*((1/h)*(-1/h))

        def Integrand_kjj(ksi, h):
            return (10**5)*(1+2*ksi)*((-1/h)**2)

        kii = np.around(quad(Integrand_kii, x, x+h, args=(h))[0], decimals=4)
        kij = np.around(quad(Integrand_kij, x, x+h, args=(h))[0], decimals=4)
        kjj = np.around(quad(Integrand_kjj, x, x+h, args=(h))[0], decimals=4)

        kel = np.array([[kii, kij],
                        [kij, kjj]])

        return kel

    for i in range(0,nEl):
        kel = ElStiff(x, h)
        Ce = np.array([int(elCon[i,0]),int(elCon[i,1])])

        for j in range(0,2):
            for z in range(0,2):
                Kglobal[Ce[j],Ce[z]] = Kglobal[Ce[j],Ce[z]]+kel[j,z]
        x=x+h

    print('Global Stiffness Matrix is constructed succesfully!\n')
    return Kglobal

"""Construction of the global force vector"""

def GlobalForce(h, L, f):

    x = 0
    nEl = int(L/h) # Number of elements
    nDOF = nEl+1   # Number of DOFs
    DOFs = np.arange(0,nDOF,1)
    Fglobal = np.zeros((nDOF, 1)) # Global force vector
    elCon = np.zeros((nEl,2))     # Element connectivity matrix

    for i in range(0,len(elCon)):
        for j in range(0,len(elCon.T)):
            elCon[i,j] = i+j

    def ElForce(f, h):

        def Integrand_fi(ksi,h):
            return f*(ksi/h)

        def Integrand_fj(ksi,h):
            return f*(1-ksi/h)

        Fi = np.around(quad(Integrand_fi, 0, 0+h, args=(h))[0], decimals=4)
        Fj = np.around(quad(Integrand_fj, 0, 0+h, args=(h))[0], decimals=4)

        Fel = np.array([[Fi],
                       [Fj]])

        return Fel

    for i in range(0,nEl):
        Fel = ElForce(f, h)
        Cf = np.array([int(elCon[i,0]),int(elCon[i,1])])

        for j in range(0,2):
            Fglobal[Cf[j]] = Fglobal[Cf[j]]+Fel[j]

        x=x+h

    if sum(Fglobal)==f:
        print('The summation of the global force vector entries equals to the body force!\n')
        print('Global Force Vector is constructed succesfully!\n')
    else:
        print('The summation of the global force vector entries is not equal to the body force!\n Please check for a potential bug!')

    return Fglobal

"""Solving the linear system to obtain dispalcements   """

def FiniteElementSolver(h, L, f, T, DBCStart, DBCEnd, u0, uL, dx):

    Kglobal = GlobalStiff(h, L)
    Fglobal = GlobalForce(h, L, f)

    if T>0:
        Fglobal[-1] = Fglobal[-1]+T
        print('Traction force is added to the last nodal point!\n')

    if DBCStart==1:
        Kf  = Kglobal[1::,1::]
        Ke  = Kglobal[0:1,0:1]
        Kef = Kglobal[0:1,1::]
        De  = np.array([u0])

        Ff = Fglobal[1::]

    if DBCEnd==1: # DOF numbering should be adjusted accordingly!
    # However, for this specific example, no DOF numbering manipulation is performed.
    # Therefore, this block does not work.

        Kf  = Kglobal[2::,2::]
        Ke  = Kglobal[0:2,0:2]
        Kef = Kglobal[0:2,2::]
        De  = np.array([u0, uL])

        Ff = Fglobal[2::]

    u = np.dot(np.linalg.inv(Kf),(Ff-Kef.T*De))

    u = np.insert(u, 0, u0)
    print('Linear system is solved for displacements!\n')

    def InterpolatedResponse(u,h,L,dx):

        x = np.arange(0, L+h,h)

        xInterpolated = np.arange(0, L+dx,dx)

        uInterpolated = np.interp(xInterpolated, x, u)

        return x, xInterpolated, uInterpolated

    x, xInterpolated, uInterpolated = InterpolatedResponse(u,h,L,dx)
    print('Interpolated results are generated!\n')

    uPrime = np.diff(uInterpolated)/dx

    return u, x, xInterpolated, uInterpolated, uPrime

"""Exact solution"""

def ExactSolution(dx):

    xValues = np.arange(0,L+dx,dx)
    uExact = (0.0575)*np.log(2*xValues+1)-0.005*xValues
    uExactPrime = (0.115)/(2*xValues+1)-0.005

    return uExact, uExactPrime

def ExactAtPoints(points):
    points = np.array(points, dtype=float)
    uExactPts = (0.0575)*np.log(2*points+1)-0.005*points
    uExactPrimePts = (0.115)/(2*points+1)-0.005
    return uExactPts, uExactPrimePts

def EvaluateFEMAtPoints(u, h, L, points):
    xNodes = np.arange(0, L+h, h)
    uAtPts = np.interp(points, xNodes, np.asarray(u).flatten())

    # Derivative for linear 1D FEM is piecewise constant in each element.
    uFlat = np.asarray(u).flatten()
    elementSlopes = np.diff(uFlat)/h
    duDxAtStart = elementSlopes[0]
    duDxAtEnd = elementSlopes[-1]

    return uAtPts, duDxAtStart, duDxAtEnd

def PrintPointwiseTable(results):
    header = (
        f"{'h':>10} | {'u_h(1/3)':>14} | {'u_exact(1/3)':>14} | "
        f"{'u_h(1.0)':>14} | {'u_exact(1.0)':>14} | "
        f"{'u_h_prime(0)':>14} | {'u_exact_prime(0)':>14} | "
        f"{'u_h_prime(1.0)':>14} | {'u_exact_prime(1.0)':>14}"
    )
    print("\nPointwise comparison table")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['h']:10.5f} | "
            f"{row['u_h_1_3']:14.6e} | {row['u_exact_1_3']:14.6e} | "
            f"{row['u_h_1_0']:14.6e} | {row['u_exact_1_0']:14.6e} | "
            f"{row['uh_prime_0']:14.6e} | {row['u_exact_prime_0']:14.6e} | "
            f"{row['uh_prime_1_0']:14.6e} | {row['u_exact_prime_1_0']:14.6e}"
        )

    print("-" * len(header))

"""Calculation of L2 Norm and Energy Norm errors"""

def ErrorCalc(xInterpolated, uExact, uInterpolated, uExactPrime, uPrime):

    # L2 norm is evaluated on the interpolation grid.
    error_u = uExact - uInterpolated
    errorL2 = np.sqrt(np.trapezoid(error_u**2, xInterpolated))

    # Energy norm uses derivative values on consistent element-wise points.
    xEnergy = xInterpolated[:-1]
    error_du = uExactPrime[:-1] - uPrime
    integrand = (10**5)*(1+2*xEnergy)*(error_du**2)
    errorEN = np.sqrt(0.5*np.trapezoid(integrand, xEnergy))

    return errorL2, errorEN

"""Call functions"""

uExact, uExactPrime = ExactSolution(dx)
hList = np.array([1/4,1/8,1/16,1/32])

errorL2 = []
errorEN = []
pointwiseResults = []

uList = []
xList = []
uPrimeList = []
xInterpolatedList = []
uInterpolatedList = []

for i in range(len(hList)):
    print(f'============ Analysis Started for h={hList[i]} ============')
    u, x, xInterpolated, uInterpolated, uPrime = FiniteElementSolver(hList[i], L, f, T, DBCStart, DBCEnd, u0, uL, dx)
    uList.append(u)
    xList.append(x)
    uPrimeList.append(uPrime)
    xInterpolatedList.append(xInterpolated)
    uInterpolatedList.append(uInterpolated)

    evalPts = np.array([1/3, 1.0])
    u_h_pts, uhPrime0, uhPrime1 = EvaluateFEMAtPoints(u, hList[i], L, evalPts)
    uExactPts, _ = ExactAtPoints([1/3, 1.0])
    _, uExactPrimeBoundary = ExactAtPoints([0.0, 1.0])

    pointwiseResults.append({
        'h': hList[i],
        'u_h_1_3': u_h_pts[0],
        'u_exact_1_3': uExactPts[0],
        'u_h_1_0': u_h_pts[1],
        'u_exact_1_0': uExactPts[1],
        'uh_prime_0': uhPrime0,
        'u_exact_prime_0': uExactPrimeBoundary[0],
        'uh_prime_1_0': uhPrime1,
        'u_exact_prime_1_0': uExactPrimeBoundary[1]
    })

    print(f'=========== Analysis Completed for h={hList[i]} ===========')
    print('============================================================\n')

    errL2, errEN = ErrorCalc(xInterpolated, uExact, uInterpolated, uExactPrime, uPrime)
    errorL2.append(errL2)
    errorEN.append(errEN)

"""Generate necessary plots"""

def uPlot(uInterpolatedList, dx, uExact, L, hList):

    xValues = np.arange(0,L+dx,dx)

    cm = 0.393701
    plt.figure(999, figsize=(15*cm, 8*cm), dpi=600)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.plot(xValues, uExact, color='black', linestyle='solid')
    plt.plot(xValues, uInterpolatedList[0], linestyle='dotted')
    plt.plot(xValues, uInterpolatedList[1], linestyle='--')
    plt.plot(xValues, uInterpolatedList[2], linestyle='-.')
    plt.plot(xValues, uInterpolatedList[3], linestyle=':')

    plt.xlim([min(xValues), max(xValues)])

    plt.xlabel('Length, in meters')
    plt.ylabel('Displacement, in meters')
    plt.title('Displacement Response')
    plt.grid()
    plt.legend(['Exact Sol.', 'h=1/4', 'h=1/8', 'h=1/16', 'h=1/32'])
    plt.savefig("uResponse.pdf")

def uPrimePlot(uPrimeList, dx, uExactPrime, L, hList):

    xValues = np.arange(0,L+dx,dx)

    cm = 0.393701
    plt.figure(998, figsize=(15*cm, 8*cm), dpi=600)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.plot(xValues, uExactPrime, color='black', linestyle='solid')
    plt.plot(xValues[:-1], uPrimeList[0], linestyle='dotted')
    plt.plot(xValues[:-1], uPrimeList[1], linestyle='--')
    plt.plot(xValues[:-1], uPrimeList[2], linestyle='-.')
    plt.plot(xValues[:-1], uPrimeList[3], linestyle=':')

    plt.xlim([min(xValues), max(xValues)])

    plt.xlabel('Length, in meters')
    plt.ylabel('The derivative of the displacement')
    plt.title('The First Derivative of the Displacement Response')
    plt.grid()
    plt.legend(['Exact Sol.', 'h=1/4', 'h=1/8', 'h=1/16', 'h=1/32'])
    plt.savefig("uPrimeResponse.pdf")

def ErrorPlot(hList, errorL2, errorEN):
    cm = 0.393701

    log_h = np.log(hList)
    log_errorL2 = np.log(errorL2)
    slopeL2, interceptL2 = np.polyfit(log_h, log_errorL2, 1)
    fitL2_log = slopeL2*log_h + interceptL2

    log_errorEN = np.log(errorEN)
    slopeEN, interceptEN = np.polyfit(log_h, log_errorEN, 1)
    fitEN_log = slopeEN*log_h + interceptEN

    plt.figure(997, figsize=(15*cm, 8*cm), dpi=600)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.plot(log_h, log_errorL2, color='black', linestyle='solid',
             marker='o', markerfacecolor='darkred', markeredgecolor='darkred',
             label='L2 data')
    plt.plot(log_h, fitL2_log, color='darkblue', linestyle='--',
             label=f'L2 fit slope = {slopeL2:.4f}')

    plt.plot(log_h, log_errorEN, color='darkgreen', linestyle='solid',
             marker='s', markerfacecolor='darkgreen', markeredgecolor='darkgreen',
             label='Energy data')
    plt.plot(log_h, fitEN_log, color='darkorange', linestyle='--',
             label=f'Energy fit slope = {slopeEN:.4f}')

    plt.xlabel('log(h)')
    plt.ylabel('log(error)')
    plt.title('Convergence: log(error) vs log(h)')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.savefig("ConvergenceError.pdf")

    return slopeL2, slopeEN

uPlot(uInterpolatedList, dx, uExact, L, hList)
uPrimePlot(uPrimeList, dx, uExactPrime, L, hList)
PrintPointwiseTable(pointwiseResults)
slopeL2, slopeEN = ErrorPlot(hList, errorL2, errorEN)
print(f"\nEstimated convergence rate from log(error)-log(h) fit:")
print(f"L2 norm slope     = {slopeL2:.6f}")
print(f"Energy norm slope = {slopeEN:.6f}")