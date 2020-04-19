



class LIterator:
    def __init__(self,x,y):
        self.x = x
        self.y = y 
        
    def __iter__(self):
        for x in range(0,min(self.x,self.y)):
            for y in range(min(x+1,self.y)):
                yield(x,y)
            for x_ in range(1,y+1):
                yield(x-x_,y)
        if self.x > self.y:
            for x_ in range(self.y,self.x):
                for y_ in range(self.y):
                    yield(x_,y_)
        elif self.y > self.x:
            for y_ in range(self.x,self.y):
                for x_ in range(1,self.x+1):
                    yield(self.x-x_,y_)
                
                
                
def write_conv_kernel_to_image_summaries(writer,tag,kernel,step,image_transform=lambda x:x,max_summaries=8):
    for n,(i,j) in enumerate(LIterator(*(kernel.shape[-2:]))):
        if n >= max_summaries:
            break
        writer.add_image(tag+'[%d,%d]'%(i,j),(image_transform(kernel[i,j])).unsqueeze(0),step)
        
        

        
        
        
        
        