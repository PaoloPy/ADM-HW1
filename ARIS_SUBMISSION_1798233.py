#CODES FOR THE DIFFERENT EXERCISES (HOMEWORK ORDER)
#I USED PYPY FOR UNKOWN REASONS (DIDN'T KNOW I COULD SWITCH) UNTIL HALFWAY THROUGH

#1  -- Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

#2  -- Python If-Else
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(raw_input().strip())
    if (n % 2) != 0:
        print("Weird")
    else:
        if (n > 5) and (n < 21):
            print("Weird")
        else:
            print("Not Weird")

#3 -- Arithmetic Operators
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)

#4 Python: Division
from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a//b)
    print(a/b)


#5 Loops
if __name__ == '__main__':
    n = int(raw_input())
    for elem in range(0,n):
        print(elem**2)


#6 Write a function
def is_leap(year):
    leap = False
    if (year % 4) == 0:
        leap = True
        if (year % 100)== 0:
            if (year % 400) != 0:
                leap = False
                
    
    # Write your logic here
    
    return leap


#7 Print Function
from __future__ import print_function

def funzione(n):
    res = ""
    for elem in range(1,n+1):
        res += str(elem)
    return res
        
if __name__ == '__main__':
    n = int(raw_input())
    print(funzione(n))


#8 Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    massimo = max(arr)
    trovato = -1000000
    for elem in arr:
        if elem != massimo and elem > trovato:
            trovato = elem
    print(trovato)

 
#9 Nested Lists
if __name__ == '__main__':
    scores = []
    insiemi = []
    for _ in range(int(raw_input())):
        name = raw_input()
        score = float(raw_input())
        if score not in scores:
            scores.append(score)
        lista = [name,score]
        insiemi.append(lista)
    scores.sort()
    secondo_minimo = scores[1]
    risultato = []
    trovato = True
    for coppia in insiemi:
        if secondo_minimo in coppia:
            risultato.append(coppia[0])
    risultato.sort()
    for elem in risultato:
        print(elem)

        
#10 Finding the percentage
from __future__ import division
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    for name in student_marks:
        if name == query_name:
            stuff = student_marks[name] 
    lunghezza = len(stuff)
    lunghezza = float(lunghezza)
    res = float(sum(stuff)/lunghezza)
    format_float = "{:.2f}".format(res)
    print(format_float)


#11 Tuples
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t = tuple(integer_list)
    res = hash(t)
    print(res)


#12 Lists
if __name__ == '__main__':
    N = int(raw_input())
    l = []
    for _ in range(N):
        s = raw_input().split()
        cmd = s[0]
        args = s[1:]
        if cmd !="print":
            cmd += "("+ ",".join(args) +")"
            eval("l."+cmd)
        else:
            print l

    
#13 List Comprehensions
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    print ([[a,b,c] for a in range(0,x+1) for b in range(0,y+1) for c in range(0,z+1) if    a + b + c != n ])

#14 sWAP cASE
def swap_case(s):
    res = ""
    for elem in s:
        if elem.islower():
            res += elem.upper()
        else:
            res += elem.lower()
    return res
            

#15 String Split and Join
def split_and_join(line):
    line = line.replace(" ","-")
    return line

#16 What's Your Name?
def print_full_name(first, last):
    print "Hello "+first+" "+last+"! You just delved into python."

#17 Mutations
def mutate_string(string, position, character):
    res = ""
    for elem in range(0,len(string)):
        if elem == position:
            res += character
        else:
            res += string[elem]
    return res

#18 Find a string
def count_substring(string, sub_string):
    cont = 0
    lunghezza = len(sub_string)
    for elem in range(0,len(string)):
        if string[elem:lunghezza] == sub_string:
            cont += 1
        lunghezza += 1
    return cont

#19 String Validators
if __name__ == '__main__':
    s = raw_input()
    for giri in range(0,5):
        if giri == 0:
            trovato = False
            for elem in s:
                if elem.isalnum():
                    trovato = True
            print(trovato)
        if giri == 1:
            trovato = False
            for elem in s:
                if elem.isalpha():
                    trovato = True
            print(trovato)
        if giri == 2:
            trovato = False
            for elem in s:
                if elem.isdigit():
                    trovato = True
            print(trovato)
        if giri == 3:
            trovato = False
            for elem in s:
                if elem.islower():
                    trovato = True
            print(trovato)
        if giri == 4:
            trovato = False
            for elem in s:
                if elem.isupper():
                    trovato = True
            print(trovato)

#20 Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


#21 The Minion Game
def minion_game(string):
    vocali = ["A","E","I","O","U"]
    usati_k = []
    usati_s = []
    k = 0
    s = 0
    for elem in range(0,len(string)):
        if string[elem] not in vocali:
            s += len(string)-elem
        else:
            k += len(string)-elem
            
    if s > k:
        print "Stuart "+ str(s)
    if k > s:
        print "Kevin "+ str(k)
    if k == s:
        print "Draw"

#22 String Formatting
def print_formatted(number):
    width = len("{0:b}".format(number))
    for i in xrange(1,number+1):
        print "{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i,width=width)

#23 Alphabet Rangoli
def print_rangoli(size):
    myStr = 'abcdefghijklmnopqrstuvwxyz'[0:size] 
    for i in range(size-1, -size, -1):
        x = abs(i)
        if x >= 0:
            line = myStr[size:x:-1]+myStr[x:size]
            print ("--"*x+ '-'.join(line)+"--"*x)
        
#24 Capitalize!
# Complete the solve function below.
def solve(s):
    par = s.split(" ")
    y = []
    for k in range(0,len(par)):
        c = par[k].capitalize()
        y.append(c)
    
    res = " ".join(y)
    return res

#25 Text Wrap
def wrap(string, max_width):
    partenza = 0
    res = ""
    for elem in range(len(string)//max_width+1):
        res += string[partenza:(partenza+max_width)]
        res += "\n"
        partenza += max_width
    return res

 
#27 Merge the Tools!
def merge_the_tools(string, k):
    inizio = 0
    for elem in range((len(string)//k)+1):
        res = ""
        new_string = string[inizio:inizio+k]
        inizio += k
        insieme = []
        for indice in range(0,len(new_string)):
            if new_string[indice] not in insieme:
                insieme.append(new_string[indice])
                res += new_string[indice]
        print(res)

#28 Introduction to Sets
def average(array):
    nuovo = set(array)
    return (sum(nuovo)/len(nuovo))

#29 ###NUMPY PART OUT OF ORDER BECAUSE I WANTED TO TRY IT
import numpy as np
array = raw_input().split(" ")
res = []
for elem in array:
    res.append(int(elem))
print(np.reshape(res,(3,3)))


#30
import numpy
r, c = map(int, raw_input().split())
i = numpy.array([raw_input().split() for _ in range(r)], int)
print(i.transpose())
print(i.flatten())


#31
import numpy
N, M, P = map(int, raw_input().split())
n = numpy.array([map(int, raw_input().split()) for _ in range(N)])
m = numpy.array([map(int, raw_input().split()) for _ in range(M)])
print numpy.concatenate((n, m))


#32
import numpy
d = tuple(map(int, raw_input().split()))  #had to change from MAP because needed 3dims
print(numpy.zeros((d), dtype = numpy.int))
print(numpy.ones((d),dtype = numpy.int))

#33
import numpy
d = tuple(map(int, raw_input().split()))  #had to change from MAP because needed 3dims
print(numpy.zeros((d), dtype = numpy.int))
print(numpy.ones((d),dtype = numpy.int))

#34
import numpy
d = tuple(map(int, raw_input().split()))  #had to change from MAP because needed 3dims
print(numpy.zeros((d), dtype = numpy.int))
print(numpy.ones((d),dtype = numpy.int))


#35
import numpy
print(str(numpy.eye(*map(int,raw_input().split()))).replace('1',' 1').replace('0',' 0'))

#36
import numpy
N, M = map(int, raw_input().split())
a, b = (numpy.array([raw_input().split() for _ in range(N)], dtype=int) for _ in range(2))
print numpy.add(a, b)
print numpy.subtract(a, b)
print numpy.multiply(a, b)
print numpy.divide(a, b)
print numpy.mod(a, b)
print numpy.power(a, b)

#37
import numpy
numpy.set_printoptions(sign=' ')
a = numpy.array(raw_input().split(),float)
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#38
import numpy
n, m = map(int, raw_input().split())
lista = numpy.array([raw_input().split() for _ in range(n)],int)
prodotto = numpy.sum(lista, axis = 0) 
res = int(prodotto[0])
for elem in range(1,len(prodotto)):
    res *= prodotto[elem]
print(res)


#39
import numpy
N, M = map(int, raw_input().split())
n = numpy.array([map(int, raw_input().split()) for _ in range(N)])
lista = numpy.min(n, axis = 1)
print(max(lista))


#40
import numpy

nm = raw_input().split()
n = int(nm[0])
m = int(nm[1])
arr = []
for i in range(n):
    m = list(map(int, raw_input().split()))
    arr.append(m)
arr = numpy.array(arr)

mean = numpy.mean(arr, axis = 1)
var = numpy.var(arr, axis = 0)
std = numpy.std(arr)

print(mean)
print(var)

rnd = numpy.around(std, 11)
print(rnd)


#41
import numpy
N=int(raw_input())

n=numpy.array([list(map(int,raw_input().split())) for _ in range(N)])

m=numpy.array([list(map(int,raw_input().split())) for _ in range(N)])

print(numpy.dot(n,m))


#42
def arrays(arr):
    b = numpy.array(arr,float)
    b = b[::-1]
    return b

#43
import numpy
n = numpy.array(raw_input().split(), int)
m = numpy.array(raw_input().split(), int)
print numpy.inner(n, m)
print numpy.outer(n, m)

#44
import numpy
n = list(map(float,raw_input().split()))
m = raw_input()
print(numpy.polyval(n,int(m)))

#45
import numpy
N = int(raw_input())
n = numpy.array([map(float, raw_input().split()) for _ in range(N)])
numpy.set_printoptions(legacy='1.13')
print numpy.linalg.det(n)

#46  #HERE I UNDERSTOOD I COULD CHANGE EDITOR Exceptions
f = int(input())        
for test in range(f):
    try:
        a,b = map(int,input().split()) 
        division_result = a // b
        print(division_result)
    except (ZeroDivisionError,ValueError) as e:
        print("Error Code:", e)

#47 Symmetric Difference
        a = int(input())
b = list(map(int,input().split()))
c = int(input())
d = list(map(int,input().split()))
res1=[]
for elem in b:
    if elem not in d and elem not in res1:
        res1.append(elem)
for elem in d:
    if elem not in b and elem not in res1:
        res1.append(elem)
res1.sort()
for elem in res1:
    print(elem)

#48 No Idea! Set .add()
n1,m1 = input().split()
n2 = input().split()
a = set(input().split())
b = set(input().split())
tot = 0
for elem in n2:
    if elem in a:
        tot += 1
    elif elem in b:
        tot -= 1
print(tot)

#49 
m = int(input())
lista = []
for elem in range(m):
    country = input()
    lista.append(country)
lista = set(lista)
print(len(lista))

#50  ##Here i FINALLY SWITCHED TO PYTHON 3 Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
r = int(input())
for _ in range(r):
    words = input().split()
    cmd = words[0]
    if cmd == "pop":
        s.pop()
    elif cmd == "remove":
        s.remove(int(words[1]))
    elif cmd == "discard":
        s.discard(int(words[1]))

print(sum(s)) Set .union() Operation
 
#51 Set .union() Operation
r = int(input())
set1 = set(map(int,input().split()))
f = int(input())
set2 = set(map(int,input().split()))
print(len(set1.union(set2)))


#52 Set .intersection() Operation
r = int(input())
set1 = set(map(int,input().split()))
f = int(input())
set2 = set(map(int,input().split()))
print(len(set1.intersection(set2)))


#53 Set .difference() Operation
r = int(input())
set1 = set(map(int,input().split()))
f = int(input())
set2 = set(map(int,input().split()))
print(len(set1.difference(set2)))

#54 Set .symmetric_difference() Operation
r = int(input())
set1 = set(map(int,input().split()))
f = int(input())
set2 = set(map(int,input().split()))

print(len(set1.symmetric_difference(set2)))


#55 The Captain's Room
k = int(input())
lista = list(map(int,input().split()))
families = len(lista)//k
camere = len(lista)//families
d = {}
for elem in lista:
    if elem in d:
        d[elem] += 1
    else:
        d[elem] = 1
for elem in d:
    if d[elem] != camere:
        print(elem)

#56 Check Subset
for elem in range(int(input())):
    a = int(input())
    set1 = set(map(int,input().split()))
    b = int(input())
    set2 = set(map(int,input().split()))
    check = True
    for elem in set1:
        if elem not in set2:
            check = False
    print(check)

#57 Check Strict Superset
set1 = set(map(int,input().split()))
r = int(input())
check = True
for elem in range(r):
    setcheck = set(map(int,input().split()))
    if len(setcheck) == len(set1):
        check = False
    for elem in setcheck:
        if elem not in set1:
            check = False
print(check)

#58 Designer Door Mat
N, M = map(int, input().split(" "))
pattern = ".|."

for y in range(N//2):
    design = pattern *((2*(y+1))-1)
    print(design.center(M,'-'))

print("WELCOME".center(M,'-'))

for y in range(N//2-1, -1, -1):
    design = pattern *((2*(y+1))-1)
    print(design.center(M,'-'))


#59 Set Mutations
s = int(input())
set1 = set(map(int,input().split()))
r = int(input())
for _ in range(r):
    cmd, _ = input().split()
    s2 = set(map(int, input().split()))
    if(cmd == "intersection_update"):
        set1.intersection_update(s2)
    elif(cmd == "update"):
        set1.update(s2)
    elif(cmd == "symmetric_difference_update"):
        set1.symmetric_difference_update(s2)
    elif(cmd == "difference_update"):
        set1.difference_update(s2)

print(sum(set1))


#60 collections.Counter()
intero = int(input())
lista = list(map(int,input().split()))
r = int(input())
tot = 0
for elem in range(r):
    roba = input().split()
    taglia = int(roba[0])
    costo = roba[1]
    if taglia in lista:
        tot +=int(costo)
        lista.remove(taglia)
print(tot)


#61 Collections.namedtuple()
from collections import namedtuple
students, FORMAT, total = int(input()), input().split(), 0
data = namedtuple('Student', FORMAT)
for i in range(students):
    data2 = input().split()
    total += int(data(CLASS = data2[FORMAT.index('CLASS')], ID = data2[FORMAT.index('ID')], MARKS = data2[FORMAT.index('MARKS')], NAME = data2[FORMAT.index('NAME')]).MARKS) 
       
print(total/students)

#62 Collections.OrderedDict()
from collections import OrderedDict

n = int(input())

d = OrderedDict()

for i in range(n):
    fast_food = input().rsplit(' ',1)
    fast_food[-1] = int(fast_food[-1])

    if fast_food[0] in d:
        d[fast_food[0]] += fast_food[1]
    else:
        d[fast_food[0]] = fast_food[1]

for key, value in d.items():
    print(key, value)


#63 Word Order
n = int(input())
dct = {}
for i in range(n):
    word = str(input())
    if word in dct:
        dct[word] += 1
    else:
        dct[word] = 1
print(len(dct))
k = dct.values()
print(" ".join([str(x) for x in k]))

 
#64 Collections.deque()
from collections import deque
n = int(input())
k = deque()
for i in range(n):
    check = input()
    if ' ' in check: 
        operation, value = check.split()
        exec ('k'+ '.' + operation + '(' + value +')')
    else:
        exec ('k'+ '.' + check + '()')    
print(*k)

 
#65 Piling Up!
k = int(input())
for _ in range(k):
    s=int(input())
    cubes=list(map(int,input().split()))
    ls=list()
    for i in range((s//2)+1):
        x=max(cubes[i],cubes[s-1-i])
        ls.append(x)
    if sorted(ls,reverse=True)==ls:
        print("Yes")
    else:
        print("No")


#66 Company Logo
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    from collections import Counter

    chars = list(input())

    chars.sort()

    chars_collection = Counter(chars)

    for letter, frequency in chars_collection.most_common(3):
        print(letter, frequency)       

#67 DefaultDict Tutorial
from collections import defaultdict

n, m = map(int, input().split(' '))

input1 = list()
for i in range(n):
    input1.append(input())

    input2 = list()
for i in range(m):
    input2.append(input())


d = defaultdict(list)

for i in range(n):
    d[input1[i]].append(i+1)

for i in input2:    
    if i in d:
        print(*d[i])
    else:
        print(-1)

#68 Calendar Module
import calendar

month,date,year=map(int,input().split()) 
print(calendar.day_name[calendar.weekday(year,month,date)].upper())

#69 Zipped!
m,n = map(int,input().split())
cose = []
for elem in range(n):
    cose.append(list(map(float, input().split())))

for i in range(m):
    somma = 0.0
    for elem in cose:
        somma += elem[i]
    print(somma/len(cose))

#70 Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    
    k = int(input())

    sorted_table = sorted(arr, key=lambda record: record[k])

    for item in sorted_table:
        print(*item)

#71 ginortS
s = input()
minus = []
maius = []
pari = []
dispari = []
for i in s:
    if i in 'qwertyuiopasdfghjklzxcvbnm':
        minus.append(i)
    elif i in 'QWERTYUIOPASDFGHJKLZXCVBNM':
        maius.append(i)
    else:
        if int(i) in [1, 3, 5, 7, 9]:
            dispari.append(i)
        elif int(i) in [0, 2, 4, 6, 8]:
            pari.append(i)
minus.sort()
maius.sort()
pari.sort()
dispari.sort()
for i in minus:
    print(i, end='')
for i in maius:
    print(i, end='')
for i in dispari:
    print(i, end='')
for i in pari:
    print(i, end='')


#72 Map and Lambda Function
cube = lambda x:pow(x,3)
    
def fibonacci(n):
    fb_series=[0,1]
    if n>=2:
        for i in range(2,n):
            next_element=fb_series[i-2]+fb_series[i-1]
            fb_series.append(next_element)
    return (fb_series[0:n])

#73 Detect Floating Point Number

import re
pattern = r"^[+-]?\d*\.\d+$"

r = int(input())

for i in range(r):
    line = input()
    if re.match(pattern, line):
        print(True)
    else:
        print(False)


#74 Re.split()
regex_pattern = r",|\."	# Do not delete 'r'.

#75 Group(), Groups() & Groupdict()

import re 
import itertools
n = re.search(r"([a-zA-Z0-9]+)",input())
for k,v in itertools.groupby(n.group(1)):
    if len(list(v)) >=2 and k != "_":
        print(k)
        break
else:
    print("-1")

#76 Re.findall() & Re.finditer()
import re
s=input()
t=re.findall('[aeiou]{2,}',s,flags=re.IGNORECASE)
if t:
    for i in t:
        if not s.endswith(i):
            print(i)
else:
    print(-1)


#77 Re.start() & Re.end()
import re
s=input()
k=input()
if s.find(k) == -1:
    print((-1,-1))
else:    
 for i in range(len(s)-len(k)+1):
    if s[i:i+len(k)]==k:
        print((i,i+len(k)-1))

#78 Regex Substitution
import re


n=int(input())
string=[]
for i in range(n):
    s=input()
    s=re.sub(r"(?<=\s)\&\&(?=\s)","and",s)
    s=re.sub(r"(?<=\s)\|\|(?=\s)","or" ,s)
    print(s)


#79 Validating Roman Numerals

thousand = "(M){0,3}"
hundred  = "((D{0,1}C{0,3})|(C[MD]))"
ten      = "((L{0,1}X{0,3})|(X[CL]))"
unit     = "((V{0,1}I{0,3})|(I[XV]))"

regex_pattern = r"^" + thousand + hundred + ten + unit + "$"


#80 Validating phone numbers
import re
pattern = r'^(9|7|8)+\d{9}$'
N=int(input())
for i in range(N):
    S= input()
    mobile_number= re.findall(pattern,S )
    if mobile_number == [] or S.isalpha() or len(S)>10:
        print('NO')
    else:
        print('YES')

#81 Validating and Parsing Email Addresses
import re
import email.utils 

N = int(input())

pattern = r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}$'
for i in range(0, N):
    parsed_addr = email.utils.parseaddr(input())
    if re.search(pattern, parsed_addr[1]):
        print(email.utils.formataddr(parsed_addr))


#82 Hex Color Code
import re 

new = list()
for i in range(int(input())):
    code = input()
    pattern = r"(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6})(?=;|,|\))"
    s =re.findall(pattern,code)
    if s:
        new.extend(s)
for i in new:
    print(i)

#83 HTML Parser - Part 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        if attrs:
            for val in attrs:
                print("->",val[0],">",val[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        if attrs:
            for val in attrs:
                print("->",val[0],">",val[1])
                
parser = MyHTMLParser()
n = int(input())
if n >= 1:
    res = ''.join(input().strip() for _ in range(n))
parser.feed(res)


#84 HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        if "\n" not in list(data):
            print(">>> Data")
            print(data)
    def handle_comment(self, data):
        if "\n" not in list(data):
            print(">>> Single-line Comment")
        else:
            print(">>> Multi-line Comment")
        print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
  
html = ""


#85 -- candles party

#!/bin/python3

import math
import os
import random
import re
import sys


def birthdayCakeCandles(candles):
    d = {}
    for elem in(candles):
        if elem not in d:
            d[elem] = 1
        else:
            d[elem] += 1
    return(max(d.values()))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


#86 Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if (x1<=x2 and v1>=v2) or (x1>=x2 and v1<=v2):
        i = 0
        for i in range(0,20000):
            if x1 + v1*i == x2 + v2*i:
                return "YES"
    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#87 Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    cond = 5  
    res = 0
    for i in range(n):
        reach = (cond// 2)
        res += reach
        cond = reach * 3
    return res

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#88 Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        
        for elem in attrs:
            print("->", elem[0], ">", elem[1])
     
    
parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

#89 Recursive Digit Sum
import math
import os
import random
import re
import sys

def superDigit(n, k):
    sum = 0
    if n in [1,2,3,4,5,6,7,8,9]:
        return n
    else:
        for elem in str(n):
           sum += int(elem)
        sum *= k
    return superDigit(sum, 1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#90 Insertion Sort - Part 1
#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    for i in range(1, n):
        copia = arr[i]
        j = i
        while j > 0 and arr[j-1] > copia:
            arr[j] = arr[j-1]
            j = j - 1
            print(*arr)
        arr[j] = copia
    print(*arr)
        


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#91 Insertion Sort - Part 2
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(0, len(arr) - 1):
        if arr[i + 1] < arr[i]:
            t = arr[i + 1]
            while i >= 0 and t < arr[i]:
                arr[i + 1] = arr[i]
                i -= 1
            arr[i + 1] = t
        print(" ".join(map(str,arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


#92 Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime as dt
# Complete the time_delta function below.
def time_delta(t1, t2):
    t_seconds = list(map(lambda t: int(dt.strptime(t, '%a %d %b %Y %X %z') .timestamp()), [t1, t2]))
        
    return str(abs(t_seconds[0] - t_seconds[1]))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#93 Validating UID
import re
pattern = r'^(?=(.*[A-Z]){2})(?=(.*\d){3})(?!.*(.).*\3)\w{10}$'
for _ in range(int(input())):
    r = re.search(pattern,input())
    if r : 
        print('Valid')
    else: 
        print('Invalid')


#94 Validating Credit Card Numbers
import re 
pattern = r'^(?!.*(\d)(-?\1){3})[456]\d{3}(?:-?\d{4}){3}$'
for _ in range(int(input())):
    print('Valid' if re.search(pattern,input()) else 'Invalid')


#95 Validating Postal Codes
regex_integer_in_range = r'^[100000-999999]{6}$'
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

 
#96 Matrix Script
import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

import re
pattern = re.compile(r'([a-zA-Z0-9])[!@#$%&) ]+([a-zA-Z0-9])')
s = ''
for j in range(len(matrix[0])):
    for i in range(len(matrix)):
        s += matrix[i][j]
        
s = pattern.sub(lambda obj: f'{obj.group(1)} {obj.group(2)}', s)
print(s)

#97 XML 1 - Find the Score
def get_attr_number(node):
    return len(node.attrib) + sum(get_attr_number(child) for child in node)

#98 XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    for child in elem:
        depth(child, level)
    maxdepth = max(level,maxdepth)
    return maxdepth 

#99 Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

#100 Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

#101 Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        return [f(p) for p in sorted(people, key=lambda x: int(x[2]))]
    return inner




