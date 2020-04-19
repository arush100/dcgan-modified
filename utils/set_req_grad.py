def set_req_grad(module,mode):
    for p in module.parameters():
        p.requires_grad = mode