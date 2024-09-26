# Task 1

# inp = input("Enter a String:")

# arr = inp.split(" ")

# for i in range(len(arr)):
#     word = arr[i]
#     first = word[0].lower()
#     if (first == 'a' or first == 'e' or first == 'i' or first == 'o' or first == 'u'):
#         arr[i] = word + "hay"
#     else:
#         arr[i] = word[1:] + word[0] + "ay"

# print(" ".join(arr))

# Task 2

# def fibonacci (length):
#     if length == 0:
#         return []
#     elif length == 1:
#         return [0]
#     elif length == 2:
#         return [0,1]
#     else:
#         arr = [0,1]
#         for i in range(2,length):
#             arr.append(arr[i-1]+arr[i-2])
#     return arr


# if __name__ == "__main__":
#     length = int(input("Enter the length of the fibonacci series:"))
#     print(fibonacci(length))

# Task 3

# def password_generator(length,count):
#     import random
#     import string
#     passwords = []
#     for i in range(count):
#         password = ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(length))
#         passwords.append(password)
#     return passwords
        

# if __name__ =="__main__":
#     length = int(input("Enter the length of the password:"))
#     count = int(input("Enter the number of passwords:"))
#     print("Random passwords: ",password_generator(length,count))

# Task 4

# class Complex:
#     def __init__(self,real,imaginary):
#         self.real = real
#         self.imaginary = imaginary

#     def __magnitude__ (self):
#         return (self.real**2 + self.imaginary**2)**0.5
    
#     def __orientation__ (self):
#         import math
#         return math.atan(self.imaginary/self.real)
    

# if __name__ == "__main__":
#     real = int(input("Enter the real part of the complex number:"))
#     imaginary = int(input("Enter the imaginary part of the complex number:"))
#     complex_number = Complex(real,imaginary)
#     print("Magnitude of the complex number:",complex_number.__magnitude__())
#     print("Orientation of the complex number:",complex_number.__orientation__())

# Task 5

class Tree():
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

    def __insert__ (self,data):
        if self.data == None:
            self.data = data
        else:
            if data < self.data:
                if self.left == None:
                    self.left = Tree()
                    self.left.data = data
                else:
                    self.left.__insert__(data)
            else:
                if self.right == None:
                    self.right = Tree()
                    self.right.data = data
                else:
                    self.right.__insert__(data)

    def __search__ (self,data):
        if self.data == data:
            return True
        elif data < self.data:
            if self.left == None:
                return False
            else:
                return self.left.__search__(data)
        else:
            if self.right == None:
                return False
            else:
                return self.right.__search__(data)

    def __print__ (self):
        if self.left:
            self.left.__print__()
        print(self.data)
        if self.right:
            self.right.__print__()
    
if __name__ == "__main__":
    len = int(input("Enter the number of elements:"))
    root = Tree()

    for i in range(len):
        element = int(input('Enter the element #' + str(i) + ": "))
        root.__insert__(element)

    print("Tree:")
    root.__print__()
    search = int(input("Enter the element to search:"))
    if root.__search__(search):
        print("Element found")
    else:
        print("Element not found")
 

    