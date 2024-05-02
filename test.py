

class A(): 

    def __init__(self): 
        print("A.__init__")

        self.setup()

    def setup(self): 
        print("A.setup")

class B(A):
    
    def __init__(self): 
        print("B.__init__")
        super().__init__()

    def setup(self): 
        print("B.setup")

b = B()