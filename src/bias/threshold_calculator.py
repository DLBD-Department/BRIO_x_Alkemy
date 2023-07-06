def threshold_calculator(A1: str, A2: int, A3: int):

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

    epsilon = ((scalA2 * scalA3) * maxi) + ((1-(scalA2 * scalA3)) * mini)

    return epsilon


