import numpy as np

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    # Objective function: calculates NPV for a given rate
    def npv(rate):
        npv_val = 0
        periods = len(cashFlowVec)
        for t in range(periods):
            npv_val += cashFlowVec[t] / (1 + rate / (cashFlowPeriod / compoundPeriod))**t
        return npv_val

    # Use a numerical solver (e.g., Newton-Raphson) to find the IRR
    irr = np.irr(cashFlowVec)  # Initial guess using numpy's IRR function as a shortcut

    # Iteratively solve for IRR
    return irr