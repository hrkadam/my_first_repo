import os

print("Hello")
ls = []
power_name = [3,4,5,6,1]
new_diff = [3,4,5,6,1]
arr_ls = [0,0,0,0]
print("HI")
print("NEW")
print("add")

ls_dec_new = [1,8,3,5,9]
new_ls_29 = [4,7,8,2,1]
print('new msg')
print("new sentense")

def fun_subt(a,b):
    return a-b
    
def new_fun(x,y,z):
    c=x+y+z
    return c

def multi(a,b):
    return a*b

def add_3(a,b,c):
    return a+b+c

def square_fun(x):
    return x*x
    
print("inside fun")

my_list = []
def win_test(a,b):
    return a+b

def max_fun(x,y,z):
    v = x*y*z
    return v


primes = []

for num in range(2, 101):   # Start from 2, since 0 and 1 are not prime
    is_prime = True
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(num)

print(primes)