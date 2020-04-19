class TrainingHook:
    def __init__(self):
        self.step = 0
        self._has_called_setup = False
        
    def __call__(self,*args,**kwargs):
        self.call(*args,**kwargs)
        self.step += 1
        
    def call(self,*args,**kwargs):
        raise NotImplementedError()
    
    def setup(self,*args,**kwargs):
        assert self.step == 0
        if self._has_called_setup:
            raise RuntimeError('.setup() has already been called on this hook.')
        self._has_called_setup = True
        return self
    
    def notify(self,msg):
        print(self.notify_str(msg))
        
    def notify_str(self,msg):
        return '[%s] %s'%(self,msg)
        
    def __repr__(self):
        try:
            return self.name
        except AttributeError:
            return self.__class__.__name__ + ' on step %d'%self.step
