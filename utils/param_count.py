def param_count(m):
    t = 0 
    for p in m.parameters():
        t += p.numel()
    return t