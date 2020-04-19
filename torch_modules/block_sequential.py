import torch
def block_sequential(module_class_list,module_kwargs_list):
    assert len(module_class_list) == len(module_kwargs_list)
    
    inner_depth = len(module_class_list)
    
    arg_lens = [(len(x) if isinstance(x,list) else 1) for x in module_kwargs_list]
    non_repeat_lens = [x for x in arg_lens if x!=1]
    if len(non_repeat_lens):
        outer_depth = non_repeat_lens[0]
        assert all([x==outer_depth for x in non_repeat_lens[1:]])
        
    for i,arg_len in enumerate(arg_lens):
        if len(arg_len) == 1:
            if isinstance(module_kwargs_list[i],list):
                module_kwargs_list[i] = [module_kwargs_list[i][0] for _ in range(outer_depth)]
            else:
                module_kwargs_list[i] = [module_kwargs_list[i] for _ in range(outer_depth)]
        
        
    return torch.nn.Sequential(*[torch.nn.Sequential(*[module_class_list[j](**module_kwargs_list[j][i]) 
                                                       for j in range(inner_depth)]) 
                                                       for i in range(outer_depth)])