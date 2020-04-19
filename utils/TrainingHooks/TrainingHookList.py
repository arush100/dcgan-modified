from utils.TrainingHooks.TrainingHook import TrainingHook


class TrainingHookList(TrainingHook):
    def __init__(self,*hooks):
        super().__init__()
        self.hooks = hooks
        
    def call(self,*args,**kwargs):
        for hook in self:
            hook(*args,**kwargs)
            
    def setup(self,*args,**kwargs):
        super().setup()
        for hook in self:
            hook.setup(*args,**kwargs)
        return self
    
    def __getitem__(self,i):
        return self.hooks[i]
    
    def __iter__(self):
        return iter(self.hooks)