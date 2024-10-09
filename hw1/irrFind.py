def cost(X, r, times, compoundPeriod):
    """Compute the NPV"""
    return sum(X[i] / (1 + r / (12/compoundPeriod)) ** (i*times) for i in range(len(X)))
    
def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    # Min diff (set at 10x smaller than the example out)
    eps = 0.000000001
    upper, lower = 0.1, -0.1
    times = int(cashFlowPeriod / compoundPeriod)
    
    while upper - lower > eps:
        midpoint = (lower + upper) / 2
        f_mid = cost(cashFlowVec, midpoint, times, compoundPeriod)

        if f_mid == 0:
            return midpoint  # Exact solution found
        elif cost(cashFlowVec, lower, times, compoundPeriod) * f_mid < 0:
            upper = midpoint  # Root is in the lower half
        else:
            lower = midpoint  # Root is in the upper half
    return (lower + upper) / 2