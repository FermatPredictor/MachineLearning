"""
程式功能:
對二維矩陣做翻轉、旋轉的基本操作，
以二維矩陣表示影像時，同時也是影像的基本操作。
相關:
(LeetCode 48題) 	Rotate Image
(LeetCode 832題) Flipping an Image
"""

class Image():
    def __init__(self, arr):
        self.arr = arr
    
    def show(self):
        for a in self.arr:
            print(a)
        print()
            
    #垂直翻轉
    def flipVertical(self):
        self.arr = self.arr[::-1]
    
    #水平翻轉
    def flipHorizontal(self):
        self.arr = [a[::-1] for a in self.arr]
    
    #行列互換(沿左上- 右下對角線翻轉)
    def rowColumnChange(self):
        self.arr = list(map(list,(zip(*self.arr))))
        
    def rotateRight(self):
        self.flipVertical()
        self.rowColumnChange()
    
    def rotateLeft(self):
        self.flipHorizontal()
        self.rowColumnChange()
    
    #顏色互補
    def colorReverse(self, color=255):
        self.arr=[[color-num for num in ar] for ar in self.arr]

if __name__ == '__main__': 
    A = [list(range(i,i+4)) for i in range(1,13,4)]
    arr= Image(A)
    arr.show()
    arr.rotateLeft()
    arr.show()
    arr.rotateRight()
    arr.show()
    arr.flipVertical()
    arr.show()
    arr.flipHorizontal()
    arr.show()
    arr.rowColumnChange()
    arr.show()
    arr.colorReverse()
    arr.show()
