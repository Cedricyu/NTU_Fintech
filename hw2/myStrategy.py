def myStrategy(pastPriceVec, currentPrice):
    import numpy as np

    # Define parameters for the moving averages and thresholds
    shortWindow = 10   # Short-term MA window size
    longWindow = 30    # Long-term MA window size
    volatilityWindow = 20 # Window size for volatility calculation
    alpha = 1.5        # Dynamic threshold multiplier
    beta = 1.5         # Dynamic threshold multiplier

    action = 0         # Default action is to hold

    # Check that we have enough data to proceed with calculations
    dataLen = len(pastPriceVec)
    if dataLen == 0:
        return action
    
    # Calculate short-term and long-term moving averages
    if dataLen < shortWindow:
        shortMA = np.mean(pastPriceVec)   # Average of entire vector if too short
    else:
        shortMA = np.mean(pastPriceVec[-shortWindow:])
        
    if dataLen < longWindow:
        longMA = np.mean(pastPriceVec)    # Average of entire vector if too short
    else:
        longMA = np.mean(pastPriceVec[-longWindow:])
    
    # Calculate volatility over the past prices
    if dataLen < volatilityWindow:
        volatility = np.std(pastPriceVec)
    else:
        volatility = np.std(pastPriceVec[-volatilityWindow:])

    # Adjust thresholds dynamically based on volatility
    buyThreshold = alpha * volatility
    sellThreshold = beta * volatility
    
    # Determine action based on short/long MA crossover and volatility-adjusted thresholds
    if (shortMA - longMA) > buyThreshold:      # Short-term MA is significantly above long-term MA
        action = 1
    elif (shortMA - longMA) < -sellThreshold:  # Short-term MA is significantly below long-term MA
        action = -1

    return action
