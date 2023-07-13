def threshold_calculator(A1: str, A2: int, A3: int, default_threshold=None):
    '''
    This function computes a parametric threshold for comparing the distances between groups
    when looking for biases in predictions.

    Args:
        A1: str ("high" or "low"), desired sensitivity of bias detection. "high" will bring to more violations. 
        A2: int, number of classes of the sensitive feature (root variable.
        A3: int, number of inputs (e.g. people) contained in the groups we are using for the distance calculation.

    Returns:
        epsilon: float, the computed threshold
    '''

    if default_threshold is not None:
        return default_threshold

    if A1=="high":
        mini = 0.005
        maxi = 0.0275
    elif A1=="low":
        mini = 0.0275
        maxi = 0.05
    else:
        raise Exception('Only "high" or "low" are allowed for parameter A1')
    
    scalA2 = 1 - 1/A2
    scalA3 = 1 - 1/A3
    epsilon = ((1 - (scalA2 * scalA3)) * maxi) + (scalA2 * scalA3 * mini)

    return epsilon


