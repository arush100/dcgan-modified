from utils.TrainingHooks.TrainingHook import TrainingHook



class PeriodicTrainingHook(TrainingHook):
    def __init__(self,period,shift=0):
        super().__init__()
        self.period = period
        self.shift = shift
        
    def __call__(self,*args,**kwargs):
        if (self.step-self.shift)%self.period == 0:
            self.call(*args,**kwargs)
        self.step += 1
        
    @classmethod
    def from_hook(cls,hook,period,shift=0):
        out = cls(period,shift)
        out.call = hook.call
    