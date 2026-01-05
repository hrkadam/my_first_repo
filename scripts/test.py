import os

print("Hello Python")

def add(a,b,c):
    v=a+b+c
    return v

def rem(a,b,c):
    v=a-b-c
    return v

print("line added")

def mul(a,b,c):
    v=a*b*c
    return v

print("multiplication function")

print("list added")
x=[4,5,6,3,2,1]
print(len(x))

def add_two(a,b):
    return a+b

ls=[4,5,6,3,2,1]
print("last line")

list_vals = [4,5,6,3,2,1]
print("final last line")

new_list = [6,2,1,0,9]
print("really fixed?")


def tiny(a, b):
    return a + b

hw_list = [4,5,6,3,2,1]
print("testing end")

temp_list=[9,6,3,2]
print("finish it!!")

def multi123(x,y):
    t=x*y
    return t

print("new date")
ls_new=[1,2,4]

def win_add(x,y,z):
    print("making addition")
    return x+y+z

print("End of script")

def rem_fuc(x,y,z):
    print("making multiplication")
    return x*y*z

print("Finally script ending")

def mul_three(x,y,z):
    print("making multiplication")
    return x*y*z

print("line 70 end")

def mul_next(x,y,z):
    print("making multiplication")
    return x*y*z

print("line 80 end")

prime_nums = []
def prime_finder(max_num):
    for num in range(2,max_num):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            prime_nums.append(num)
    return prime_nums

prime_nums_cnt = prime_finder(50)
print('Total prime numbers count = ',len(prime_nums_cnt))