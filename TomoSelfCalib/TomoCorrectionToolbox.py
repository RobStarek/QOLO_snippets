# -*- coding: utf-8 -*-
"""
Small toolbox for correcting imperfect retardances of waveplates in
polarization state tomography.
It relies on auxilliary modules
from https://github.com/RobStarek/QOLO_snippets
and scipy and numpy.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Custom helper modules from QOLO_snippets
import KetSugar as ks #low-dimensional Dirac notation helper module
import WaveplateBox as wp #waveplates jones calculus
import MaxLik as ml #maximum-likelihood reconstruction
import HammerProj as hp #helper plotting class

deg = np.pi/180.
StandardAngleTable = [
    (0        , 0),
    (45*deg   , 0),
    (22.5*deg , 0),
    (-22.5*deg, 0),
    (0*deg , 45*deg),
    (0*deg , -45*deg)
] #for tomographic projections

def GenerateProjectors(dret1=0, dret2=0, projection_state = ks.LO, AT=StandardAngleTable):
    """
    Generate array of projector matrices with given retarance errors and
    waveplate angle table AT.
    Args:
        dret1 ... retardance error of HWP
        dret2 ... retardance error of QWP
        projection_state ... eigenket of used polarizer
        AT    ... list/table with settings of waveplates, first HWP, second QWP
    Returns:
        M ... array of projector operators
    """        
    bras = [wp.WPProj(x,y, projection_state, dret1, dret2) for x,y in AT]    
    operators = [bra.T.conjugate() @ bra for bra in bras]
    return np.array(operators)


def GenerateTheoryStates(AT, dret1 = 0, dret2 = 0, input_state = ks.LO):
    """
    Generate theoretical states (in ket form) from preparation table and known 
    waveplate errors.
    Args:
        AT    ... list/table with settings of waveplates, first HWP, second QWP
        dret1 ... retardance error of HWP
        dret2 ... retardance error of QWP
        input_state ... input polarization state
    Returns:
        list of projector operators
    """
    return [wp.WPPrepare(x, y, input_state, dret1, dret2) for x,y in AT] 
    


def ImpurityScore(data, dret1=0, dret2=0, projection_state=ks.LO, AT=StandardAngleTable):
    """
    Negative minimal purity of reconstructions from given data assuming
    dret1 and dret2 waveplate retardance errors and projection angle table.
    1 + ImpurityScore = Impurity = 1 - Purity
    """ 
    MinPur = 1
    if dret1==0: #avoid with 0,0 singularity
        dret1 = 1e-12
    if dret2==0:
        dret2 = 1e-12    
    projectors = GenerateProjectors(dret1, dret2, projection_state, AT)
    for tomogram in data:
        rho = ml.Reconstruct(tomogram, projectors, 1000, 1e-9, RhoPiVect=True, Renorm=True)
        P = ks.Purity(rho)        
        if P <= MinPur:
            MinPur = P    
    return -MinPur

def InfidelityScore(rhos,AT,dx=0, dy=0, input_state = ks.LO):
    """
    Negative mean and minimal fidelity of prepared states with respect to the expected
    states. Mean and minimum are taken over provided reference states.
    Waveplate angle table AT must much the provided rhos.
    1 + InfidelityScore = Infidelity = 1 - Fidelity
    """     
    TheoryStates = GenerateTheoryStates(AT, dx, dy, input_state)
    Fmin = 1.1
    Fmean = 0
    for i, ket in enumerate(TheoryStates):
        rho = rhos[i]
        F = ks.ExpectationValue(ket, rho)
        Fmean = Fmean + F
        if F <= Fmin:
            Fmin = F
    Fmean = Fmean/len(TheoryStates)
    #return -Fmin
    return -Fmean, -Fmin    


def EstProjRetErr(projections, angle_table, projection_state=ks.LO, mode=0):
    """
    Estiamate tomograph waveplates retardance imperfections from tomographic data.
    Tomographic data must contain measurements of (quasi)-uniformly distributed
    pure states on input. At least N=12 reference states are needed for this to work.
    Order of projections must match the order in angle table.
    N ... number of measured reference states
    M ... number of projectors in single-qubit tomography
    Args:
        projections ... NxM ndarray of floats, each line contains tomogram with M entries
                        for one each of N reference states
        angle_table ... Mx2 ndarray of waveplate rotations, each line contains
                        rotation of half- and quarter-wave plates for each projection.  
        projection_state ... ket-eigenstate of used polarizer
        mode ... if 0, Melder-Nead minimization is performed, otherwise, table
                 is produced and plotted (for debug and publication purposes)
                 (to be deleted in the future)                 
    Returns:
        dhwp ... half-wave plate retardance deviation
        dqwp ... quarter-wave plate retardance deviation    
        S ... optimal impurity score   
    """
    
    if mode==0: #Nelder-Mead optimization
        def minim(x):           
            S = ImpurityScore(projections, x[0], x[1], projection_state, angle_table)
            return S
        R = minimize(minim, (1*deg,1*deg), method='Nelder-Mead')
        return R['x'][0], R['x'][1], R['fun']
    
    else: #Produce table
        nx = 41
        ny = 41
        search_space_x = np.linspace(-10*deg,10*deg,nx)
        search_space_y = np.linspace(-10*deg,10*deg,ny)
        table = np.zeros((ny,nx))
        score_min = 1
        argmin_score = (0,0)
        for i,x in enumerate(search_space_x):
            print(i)
            for j, y in enumerate(search_space_y):
                score = ImpurityScore(projections, dret1=x, dret2=y, projection_state=projection_state, AT=angle_table)        
                table[i,j] = score
                if score < score_min:
                    score_min = score
                    argmin_score = (x,y)
        plt.matshow(table, extent=(search_space_y[0]/deg,search_space_y[-1]/deg,search_space_x[-1]/deg, search_space_x[0]/deg))
        plt.hlines([0, argmin_score[0]/deg],search_space_y[0]/deg,search_space_y[-1]/deg)
        plt.vlines([0, argmin_score[1]/deg],search_space_x[0]/deg,search_space_x[-1]/deg)
        plt.colorbar()
        plt.xlabel("$\\delta \\Gamma_2$ (QWP)")
        plt.ylabel("$\\delta \\Gamma_1$ (HWP)")
        plt.show()                    
        return argmin_score[0], argmin_score[1], score_min


def EstPrepRetErr(rhos, angle_table, input_state=ks.LO, mode=0):
    """
    Estiamate preparation waveplates retardance imperfections from tomographic data.
    Tomographic data must contain measurements of (quasi)-uniformly distributed
    pure states on input. Order of projections must match the order in angle table.    
    N ... number of measured reference states
    M ... number of projectors in single-qubit tomography
    Args:
        rhos ... Nx2x2 ndarray of reconstructed density matrices
        angle_table ... Mx2 ndarray of waveplate rotations, each line contains
                        rotation of half- and quarter-wave plates for each projection.                   
        input_state ... input polarization statate
        mode ... if 0, Melder-Nead minimization is performed, otherwise, table        
                 is produced and plotted (for debug and publication purposes)
                 (to be deleted in the future)
    Returns:
        dhwp ... half-wave plate retardance deviation
        dqwp ... quarter-wave plate retardance deviation    
        S ... optimal infidelity score      
    """
    if mode==0:
        def minim(x):        
            return InfidelityScore(rhos,angle_table, x[0],x[1], input_state)[0]
        R = minimize(minim, (1*deg, 1*deg), method='Nelder-Mead')
        return R['x'][0], R['x'][1], R['fun']

    else:
        nx = 41
        ny = 41
        search_space_x = np.linspace(-10*deg,10*deg,nx)
        search_space_y = np.linspace(-10*deg,10*deg,ny)
        table = np.zeros((ny,nx))
        minS = 1
        argminS = (0,0)
        for i,x in enumerate(search_space_x):
            print(i)
            for j, y in enumerate(search_space_y):
                #score = InfidelityScore(rhos,angle_table,x, y)[0]
                score = InfidelityScore(rhos,angle_table, x,y, input_state)[0]
                if score <= minS:
                    minS = score
                    argminS = (x,y)
                table[i,j] = score      
        
        plt.matshow(table, extent=(search_space_x[0]/deg,search_space_x[-1]/deg,search_space_y[-1]/deg, search_space_y[0]/deg))
        plt.hlines([0,argminS[0]/deg], search_space_x[0]/deg, search_space_x[-1]/deg)
        plt.vlines([0,argminS[1]/deg], search_space_y[0]/deg,search_space_y[-1]/deg)        
        plt.colorbar()
        plt.show()  
        ths0 = GenerateTheoryStates(angle_table, 0, 0, input_state)
        ths1 = GenerateTheoryStates(angle_table, argminS[0], argminS[1], input_state)
        hammer0 = []
        for ket in ths0:
            thetaM, phiM = hp.RhoToBloch(ks.ketbra(ket, ket))
            X, Y = hp.Hammer(np.pi/2 - thetaM, phiM)
            hammer0.append((X,Y))
        hammer0 = np.array(hammer0)
        hammer1 = []
        for ket in ths1:
            thetaM, phiM = hp.RhoToBloch(ks.ketbra(ket, ket))
            X, Y = hp.Hammer(np.pi/2 - thetaM, phiM)
            hammer1.append((X,Y))
        hammer1 = np.array(hammer1)
        fig, ax = plt.subplots(1,1)
        hp.PlotHammerGrid(ax, 17,17)
        hp.PlotHammerRhos(ax, rhos, "o", clr='black', label="Measured")
        hp.PlotHammerKets(ax, ths0, "o", clr='green', label="Expected")
        hp.PlotHammerKets(ax, ths1, ".", clr='orange', label="Expected w errors")
        plt.legend()
        plt.show()
        return argminS[0], argminS[1], minS

def GetCalibratedAngles(dret1=0, dret2=0, angle_table_proj=StandardAngleTable, projection_state=ks.LO):
    ProjWanted = GenerateProjectors(0, 0, projection_state, angle_table_proj)
    bras = [wp.WPProj(x,y, projection_state, dret1, dret2) for x,y in angle_table_proj]
    kets = [bra.T.conjugate() for bra in bras]    
    corr_angle_table = []
    for ket in kets:
        R = wp.SearchForProj(ket, projection_state, dret1, dret2, 1e-12)
        corr_angle_table.append(R['x'])
    corr_angle_table = np.array(corr_angle_table)
    return corr_angle_table
    
def CalibrateTomography(projections, angle_table_proj, angle_table_prep, input_state, proj_state, mode=0):
    """
    1) estimate projection waveplates errors
    2) reconstruct data with corrected projectors
    3) use reconstruction to find errors of preparation waveplates
    """
    MIT = 1000
    TRH = 1e-9
    #estimate projection waveplates retardances 
    print("Estimating projection errors...")
    dhwpA, dqwpA, SP = EstProjRetErr(projections, angle_table_proj, proj_state, mode)
    #and use them to update our knowledge of measurement operators
    CorrectedProjectors = GenerateProjectors(dhwpA, dqwpA, proj_state, angle_table_proj)
    #reconstruct reference states with corrected projectors
    print("Reconstructing references...")
    references = [ml.Reconstruct(tomogram, CorrectedProjectors, MIT, TRH, RhoPiVect=True, Renorm=True) for tomogram in projections] 
    references = np.array(references)

    #use reference states to reveal retardance erros in preparation
    print("Fitting preparation errors...")
    dhwpP, dqwpP, SF  = EstPrepRetErr(references, angle_table_prep, input_state, mode)   

    #calculate projection table with errors taken into account
    new_proj_table = GetCalibratedAngles(dhwpA, dqwpA, angle_table_proj, proj_state)

    resultDict = {
        "ImpurityScore" : SP,
        "InfidelityScore" : SF,
        "ProjRetErr" : np.array((dhwpA, dqwpA)),
        "PrepRetErr" : np.array((dhwpP, dqwpP)),
        "Projectors" : CorrectedProjectors,
        "Refs" : references,
        "CalibratedProjTable" : new_proj_table
    }
    return resultDict




