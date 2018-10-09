
# coding: utf-8

# # Function for Keypoint Detection
# 

# In[ ]:


import cv2
import sys
import math


# In[ ]:


def function(sigma):
    num=np.asarray([[0.0 for x in range(0,7)] for y in range(0,7)])
    s=0
    for x in range(0,7):
        for y in range(0,7):
            
            value1=x-3.0
            value2=3.0-y
            a = ( value1 ** 2.0 + value2 ** 2.0) / (2.0 * (sigma ** 2.0))
            exp = math.exp(-(a))
            d = 2.0 * 3.14 * (sigma ** 2.0)
            t = exp / d
            s=s+t
            num[x][y]=t
            
    for x in range(0,7):
        for y in range(0,7):
            num[x][y]=num[x][y]/s
            
        
            
            
            #print(num)
    return (num)


# In[3]:


def resize(img):
    rimage=img[0:len(img):2,0:len(img[0]):2]
    return rimage


# In[4]:


a=1.0/(2.0**(0.5))
b=1.0
c=(2.0**(0.5))
d=2.0
e=2.0*(2.0**(0.5))

f=(2.0**(0.5))
g=2.0
h=2.0*(2.0**(0.5))
i=4.0
j=4.0*(2.0**(0.5))

k=2.0*(2.0**(0.5))
l=4.0
m=4.0*(2.0**(0.5))
n=8.0
o=8.0*(2.0**(0.5))

p=4.0*(2.0**(0.5))
q=8.0
r=8.0*(2.0**(0.5))
s=16.0
t=16.0*(2.0**(0.5))



gaussian1=function(a)
gaussian2=function(b)
gaussian3=function(c)
gaussian4=function(d)
gaussian5=function(e)

gaussian6=function(f)
gaussian7=function(g)
gaussian8=function(h)
gaussian9=function(i)
gaussian10=function(j)

gaussian11=function(k)
gaussian12=function(l)
gaussian13=function(m)
gaussian14=function(n)
gaussian15=function(o)

gaussian16=function(p)
gaussian17=function(q)
gaussian18=function(r)
gaussian19=function(s)
gaussian20=function(t)

#print(gaussian1)
#print(gaussian2)
#print(gaussian3)
#print(gaussian4)
#print(gaussian5)


# In[5]:


img= cv2.imread('/Users/krishna/Downloads/proj1_cse573-5/task2.jpg', 0)
value=(img)


# In[6]:


def pad(values):
        row=len(values)
        col=len(values[0])
        #print(row,col)
        
        padding=np.array([[0 for i in range(col+6)] for j in range(row+6)])  
        
        for x in range(3, row+3): 
            for y in range(3, col+3):
                padding[x, y]=values[x-3, y-3]
        return(padding)


# In[7]:


def func(image,kernel):
    row= image.shape[0]
    col= image.shape[1]
    #print(row, col)
    edged=np.array([[0 for i in range(col)] for j in range(row)]) 
    #edged = np.zeros(value.shape)
    for i in range(3, row-3):
        for j in range(3,col-3):
            z=kernel[0][0]*image[i-3][j-3]+kernel[0][1]*image[i-3][j-2]+kernel[0][2]*image[i-3][j-1]+kernel[0][3]*image[i-3][j]+kernel[0][4]*image[i-3][j+1]+kernel[0][5]*image[i-3][j+2]+kernel[0][6]*image[i-3][j+3]+kernel[1][0]*image[i-2][j-3]+kernel[1][1]*image[i-2][j-2]+kernel[1][2]*image[i-2][j-1]+kernel[1][3]*image[i-2][j]+kernel[1][4]*image[i-2][j+1]+kernel[1][5]*image[i-2][j+2]+kernel[1][6]*image[i-2][j+3]+kernel[2][0]*image[i-1][j-3]+kernel[2][1]*image[i-1][j-2]+kernel[2][2]*image[i-1][j-1]+kernel[2][3]*image[i-1][j]+kernel[2][4]*image[i-1][j+1]+kernel[2][5]*image[i-1][j+2]+kernel[2][6]*image[i-1][j+3]+    kernel[3][0]*image[i][j-3]+kernel[3][1]*image[i][j-2]+kernel[3][2]*image[i][j-1]+kernel[3][3]*image[i][j]+kernel[3][4]*image[i][j+1]+kernel[3][5]*image[i][j+2]+kernel[3][6]*image[i][j+3]+kernel[4][0]*image[i+1][j-3]+kernel[4][1]*image[i+1][j-2]+kernel[4][2]*image[i+1][j-1]+kernel[4][3]*image[i+1][j]+kernel[4][4]*image[i+1][j+1]+kernel[4][5]*image[i+1][j+2]+kernel[4][6]*image[i+1][j+3]+kernel[5][0]*image[i+2][j-3]+kernel[5][1]*image[i+2][j-2]+kernel[5][2]*image[i+2][j-1]+kernel[5][3]*image[i+2][j]+kernel[5][4]*image[i+2][j+1]+kernel[5][5]*image[i+2][j+2]+kernel[5][6]*image[i+2][j+3]+kernel[6][0]*image[i+3][j-3]+kernel[6][1]*image[i+3][j-2]+kernel[6][2]*image[i+3][j-1]+kernel[6][3]*image[i+3][j]+kernel[6][4]*image[i+3][j+1]+kernel[6][5]*image[i+3][j+2]+kernel[6][6]*image[i+3][j+3] 
            edged[i][j]=z
            
    return(edged)


# In[8]:


def normalise(matrix):
    maximum=0
    minimum=matrix[1][1]
    pos_edge_x =[[0 for x in range(len(matrix[0]))] for y in range(len(matrix))] 
    
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]>maximum:
                maximum=matrix[i][j]

    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]<minimum:      
                minimum=matrix[i][j]
                
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            pos_edge_x[i][j] = ((matrix[i][j] - minimum) / (maximum - minimum))
    

    return(pos_edge_x)


# In[ ]:


imageoct1=np.array(resize(value))
answer1=np.array(pad(imageoct1))
imageoct2=np.array(resize(imageoct1))
answer2=np.array((imageoct2))
imageoct3=np.array(resize(imageoct2))
answer3=np.array((imageoct3))
imageoct4=np.array(resize(imageoct3))
answer4=np.array(imageoct4)


g1=np.array(func(answer1,gaussian1))
g2=np.array(func(answer1,gaussian2))
g3=np.array(func(answer1,gaussian3))
g4=np.array(func(answer1,gaussian4))
g5=np.array(func(answer1,gaussian5))

g6=np.array(func(answer2,gaussian6))
g7=np.array(func(answer2,gaussian7))
g8=np.array(func(answer2,gaussian8))
g9=np.array(func(answer2,gaussian9))
g10=np.array(func(answer2,gaussian10))

g11=np.array(func(answer3,gaussian11))
g12=np.array(func(answer3,gaussian12))
g13=np.array(func(answer3,gaussian13))
g14=np.array(func(answer3,gaussian14))
g15=np.array(func(answer3,gaussian15))

g16=np.array(func(answer4,gaussian16))
g17=np.array(func(answer4,gaussian17))
g18=np.array(func(answer4,gaussian18))
g19=np.array(func(answer4,gaussian19))
g20=np.array(func(answer4,gaussian20))

#print(f)

t1=np.array(normalise(g1))
t2=np.array(normalise(g2))
t3=np.array(normalise(g3))
t4=np.array(normalise(g4))
t5=np.array(normalise(g5))

t6=np.array(normalise(g6))
t7=np.array(normalise(g7))
t8=np.array(normalise(g8))
t9=np.array(normalise(g9))
t10=np.array(normalise(g10))

#cv2.namedWindow('t6', cv2.WINDOW_NORMAL)
#cv2.imshow('t6',t6)   
#cv2.waitKey(0)

#cv2.namedWindow('t7', cv2.WINDOW_NORMAL)
#cv2.imshow('t7',t7)   
#cv2.waitKey(0)

#cv2.namedWindow('t8', cv2.WINDOW_NORMAL)
#cv2.imshow('t8',t8)   
#cv2.waitKey(0)

#cv2.namedWindow('t9', cv2.WINDOW_NORMAL)
#cv2.imshow('t9',t9)   
#cv2.waitKey(0)


t11=np.array(normalise(g11))
t12=np.array(normalise(g12))
t13=np.array(normalise(g13))
t14=np.array(normalise(g14))
t15=np.array(normalise(g15))

cv2.namedWindow('t11', cv2.WINDOW_NORMAL)
cv2.imshow('t11',t11)   
cv2.waitKey(0)

#cv2.namedWindow('t12', cv2.WINDOW_NORMAL)
#cv2.imshow('t12',t12)   
#cv2.waitKey(0)

#cv2.namedWindow('t13', cv2.WINDOW_NORMAL)
#cv2.imshow('t13',t8)   
#cv2.waitKey(0)

#cv2.namedWindow('t14', cv2.WINDOW_NORMAL)
#cv2.imshow('t14',t9)   
#cv2.waitKey(0)


t16=np.array(normalise(g16))
t17=np.array(normalise(g17))
t18=np.array(normalise(g18))
t19=np.array(normalise(g19))
t20=np.array(normalise(g20))


# In[ ]:


def subtract(matrix1,matrix2):
    row1= len(matrix1)
    col1= len(matrix1[0])
    row2= len(matrix2)
    col2= len(matrix2[0])
    
    for i in range(0,row2):
        for j in range(0,col2):
            matrix1[i][j]=matrix1[i][j]-matrix2[i][j]
            
    return(matrix1)


# In[ ]:


dog1=subtract(t1,t2)
print(dog1.shape)
dog2=subtract(t2,t3)
print(dog2.shape)
dog3=subtract(t3,t4)
print(dog3.shape)
dog4=subtract(t4,t5)
print(dog4.shape)

dog6=subtract(t6,t7)
print(dog6.shape)
dog7=subtract(t7,t8)
print(dog7.shape)
dog8=subtract(t8,t9)
print(dog8.shape)
dog9=subtract(t9,t10)
print(dog9.shape)

#cv2.namedWindow('octave1', cv2.WINDOW_NORMAL)
#cv2.imshow('octave1',dog6)   
#cv2.waitKey(0)

#cv2.namedWindow('octave2', cv2.WINDOW_NORMAL)
#cv2.imshow('octave2',dog7)   
#cv2.waitKey(0)

#cv2.namedWindow('octave3', cv2.WINDOW_NORMAL)
#cv2.imshow('octave3',dog8)   
#cv2.waitKey(0)

#cv2.namedWindow('octave4', cv2.WINDOW_NORMAL)
#cv2.imshow('octave4',dog9)   
#cv2.waitKey(0)

dog10=subtract(t11,t12)
print(dog10.shape)
dog11=subtract(t12,t13)
print(dog11.shape)
dog12=subtract(t13,t14)
print(dog12.shape)

dog13=subtract(t14,t15)
print(dog13.shape)
#cv2.namedWindow('octave5', cv2.WINDOW_NORMAL)
#cv2.imshow('octave5',dog10)   
#cv2.waitKey(0)

#cv2.namedWindow('octave6', cv2.WINDOW_NORMAL)
#cv2.imshow('octave6',dog11)   
#cv2.waitKey(0)

#cv2.namedWindow('octave7', cv2.WINDOW_NORMAL)
#cv2.imshow('octave7',dog12)   
#cv2.waitKey(0)

#cv2.namedWindow('octave8', cv2.WINDOW_NORMAL)
#cv2.imshow('octave8',dog13)   
#cv2.waitKey(0)

dog15=subtract(t16,t17)
print(dog15.shape)
dog16=subtract(t17,t18)
print(dog16.shape)
dog17=subtract(t18,t19)
print(dog17.shape)
dog18=subtract(t19,t20)
print(dog18.shape)


# In[ ]:


#octave 2
row2=len(dog7)
column2=len(dog7[0])
key_dog123=np.array([[0 for i in range(column2)] for j in range(row2)])

for i in range(1,(row2-1)):
    for j in range(1,(column2-1)):
        if dog7[i][j] == max(dog7[i-1][j-1], dog7[i-1][j], dog7[i-1][j+1], dog7[i][j-1], dog7[i][j], dog7[i][j+1], dog7[i+1][j-1], dog7[i+1][j], dog7[i+1][j+1]):
            if dog7[i,j] > max(dog6[i-1,j-1], dog6[i-1,j], dog6[i-1,j+1], dog6[i,j-1], dog6[i,j], dog6[i,j+1], dog6[i+1,j-1], dog6[i+1,j], dog6[i+1,j+1]):
                if dog7[i,j] > max(dog8[i-1][j-1], dog8[i-1][j], dog8[i-1][j+1], dog8[i][j-1], dog8[i][j], dog8[i][j+1], dog8[i+1][j-1], dog8[i+1][j], dog8[i+1][j+1]):
                    key_dog123[i][j] = 255
                else:
                    continue

        else: 
            if dog7[i,j] == min(dog7[i-1][j-1], dog7[i-1][j], dog7[i-1][j+1], dog7[i][j-1], dog7[i][j], dog7[i][j+1], dog7[i+1][j-1], dog7[i+1][j], dog7[i+1][j+1]):
                if dog7[i,j] < min(dog6[i-1,j-1], dog6[i-1,j], dog6[i-1,j+1], dog6[i,j-1], dog6[i,j], dog6[i,j+1], dog6[i+1,j-1], dog6[i+1,j], dog6[i+1,j+1]):
                    if dog7[i,j] < min(dog8[i-1][j-1], dog8[i-1][j], dog8[i-1][j+1], dog8[i][j-1], dog8[i][j], dog8[i][j+1], dog8[i+1][j-1], dog8[i+1][j], dog8[i+1][j+1]):
                        key_dog123[i][j] = 0


key_dog123=np.asarray(normalise(key_dog123))
cv2.imshow('key_dog123',key_dog123)                   

row3=len(dog8)
column3=len(dog8[0])
key_dog234= np.array([[0 for i in range(column3)] for j in range(row3)])

for i in range(1,(row3-1)):
    for j in range(1,(column3-1)):
        if dog8[i][j] == max(dog8[i-1][j-1], dog8[i-1][j], dog8[i-1][j+1], dog8[i][j-1], dog8[i][j], dog8[i][j+1], dog8[i+1][j-1], dog8[i+1][j], dog8[i+1][j+1]):
            if dog8[i][j] > max(dog7[i-1][j-1], dog7[i-1][j], dog7[i-1][j+1], dog7[i][j-1], dog7[i][j], dog7[i][j+1], dog7[i+1][j-1], dog7[i+1][j], dog7[i+1][j+1]):
                if dog8[i][j] > max(dog9[i-1][j-1], dog9[i-1][j], dog9[i-1][j+1], dog9[i][j-1], dog9[i][j], dog9[i][j+1], dog9[i+1][j-1], dog9[i+1][j], dog9[i+1][j+1]):
                    key_dog234[i][j] = 255
                else:
                    continue

        else:
             if dog8[i][j] == min(dog8[i-1][j-1], dog8[i-1][j], dog8[i-1][j+1], dog8[i][j-1], dog8[i][j], dog8[i][j+1], dog8[i+1][j-1], dog8[i+1][j], dog8[i+1][j+1]):
                if dog8[i][j] < min(dog7[i-1][j-1], dog7[i-1][j], dog7[i-1][j+1],dog7[i][j-1],dog7[i][j], dog7[i][j+1], dog7[i+1][j-1],dog7[i+1][j], dog7[i+1][j+1]):
                    if dog8[i][j] < min(dog9[i-1][j-1], dog9[i-1][j], dog9[i-1][j+1],dog9[i][j-1], dog9[i][j], dog9[i][j+1], dog9[i+1][j-1], dog9[i+1][j], dog9[i+1][j+1]):
                        key_dog234[i][j] = 0

key_dog234=np.asarray(normalise(key_dog234))
cv2.imshow('key_dog234',key_dog234) 

#octave 3

row4=len(dog11)
column4=len(dog11[0])
key_dog2123=np.array([[0 for i in range(column4)] for j in range(row4)])

for i in range(1,(row4-1)):
    for j in range(1,(column4-1)):
        if dog11[i][j] == max(dog11[i-1][j-1], dog11[i-1][j], dog11[i-1][j+1], dog11[i][j-1], dog11[i][j], dog11[i][j+1], dog11[i+1][j-1], dog11[i+1][j], dog11[i+1][j+1]):
            if dog11[i][j] > max(dog10[i-1][j-1], dog10[i-1][j], dog10[i-1][j+1], dog10[i][j-1], dog10[i][j], dog10[i][j+1], dog10[i+1][j-1], dog10[i+1][j], dog10[i+1][j+1]):
                if dog11[i][j] > max(dog12[i-1][j-1], dog12[i-1][j], dog12[i-1][j+1], dog12[i][j-1], dog12[i][j], dog12[i][j+1], dog12[i+1][j-1], dog12[i+1][j], dog12[i+1][j+1]):
                    key_dog2123[i][j] = 255
                else:
                    continue

        else: 
            if dog11[i][j] == min(dog11[i-1][j-1], dog11[i-1][j], dog11[i-1][j+1], dog11[i][j-1], dog11[i][j], dog11[i][j+1], dog11[i+1][j-1], dog11[i+1][j], dog11[i+1][j+1]):
                if dog11[i][j] < min(dog10[i-1][j-1], dog10[i-1][j], dog10[i-1][j+1], dog10[i][j-1], dog10[i][j], dog10[i][j+1], dog10[i+1][j-1], dog10[i+1][j], dog10[i+1][j+1]):
                    if dog7[i][j] < min(dog8[i-1][j-1], dog8[i-1][j], dog8[i-1][j+1],dog8[i][j-1], dog8[i][j], dog8[i][j+1], dog8[i+1][j-1], dog8[i+1][j], dog8[i+1][j+1]):
                        key_dog123[i][j] = 0


key_dog2123=np.asarray(normalise(key_dog2123))

cv2.namedWindow('key_dog2123', cv2.WINDOW_NORMAL)
cv2.imshow('key_dog2123',key_dog2123)   


row5=len(dog12)
column5=len(dog12[0])
key_dog2234= np.array([[0 for i in range(column3)] for j in range(row3)])

for i in range(1,(row5-1)):
    for j in range(1,(column5-1)):
        if dog12[i][j] == max(dog12[i-1][j-1], dog12[i-1][j], dog12[i-1][j+1], dog12[i][j-1], dog12[i][j], dog12[i][j+1], dog12[i+1][j-1], dog12[i+1][j], dog12[i+1][j+1]):
            if dog12[i][j] > max(dog11[i-1][j-1], dog11[i-1][j], dog11[i-1][j+1], dog11[i][j-1], dog11[i][j], dog11[i][j+1], dog11[i+1][j-1], dog11[i+1][j], dog11[i+1][j+1]):
                if dog12[i][j] > max(dog13[i-1][j-1], dog13[i-1][j], dog13[i-1][j+1], dog13[i][j-1], dog13[i][j], dog13[i][j+1], dog13[i+1][j-1], dog13[i+1][j], dog13[i+1][j+1]):
                    key_dog2234[i][j] = 255
                else:
                    continue

        else:
             if dog12[i][j] == min(dog12[i-1][j-1], dog12[i-1][j], dog12[i-1][j+1], dog12[i][j-1], dog12[i][j], dog12[i][j+1], dog12[i+1][j-1], dog12[i+1][j], dog12[i+1][j+1]):
                if dog12[i][j] < min(dog11[i-1][j-1], dog11[i-1][j], dog11[i-1][j+1], dog11[i][j-1], dog11[i][j], dog11[i][j+1], dog11[i+1][j-1], dog11[i+1][j], dog11[i+1][j+1]):
                    if dog8[i][j] < min(dog13[i-1][j-1], dog13[i-1][j], dog13[i-1][j+1], dog13[i][j-1], dog13[i][j], dog13[i][j+1], dog13[i+1][j-1], dog13[i+1][j], dog13[i+1][j+1]):
                        key_dog2234[i][j] = 0

key_dog2234=np.asarray(normalise(key_dog2234))
cv2.namedWindow('key_dog2234', cv2.WINDOW_NORMAL)
cv2.imshow('key_dog2234',key_dog2234) 
cv2.waitKey(0)

