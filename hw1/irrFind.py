def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod, tol=1e-6, max_iter=1000):
    def npv(rate):
        npv_val = 0
        periods = len(cashFlowVec)
        for t in range(periods):
            npv_val += cashFlowVec[t] / (1 + rate / (cashFlowPeriod / compoundPeriod))**t
        return npv_val

    rate = 0.1  # Initial guess for IRR
    for _ in range(max_iter):
        npv_val = npv(rate)
        npv_derivative = sum(
            -t * cashFlowVec[t] / (1 + rate / (cashFlowPeriod / compoundPeriod))**(t + 1)
            for t in range(len(cashFlowVec))
        )
        
        if abs(npv_val) < tol:
            return rate
        
        rate -= npv_val / npv_derivative  # Update using Newton-Raphson method
    
    return None  # Return None if IRR could not be found in the given iterations
