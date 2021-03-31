def encode(y):
    if y < 0.5:
        return 0
    elif y < 1.5:
        return 1
    else:
        return 2
