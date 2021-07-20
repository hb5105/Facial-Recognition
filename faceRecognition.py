#Facial recognition using KNN classification algorithm

#load the training data(numpy arrays)  
#read video stream using opencv
#use knn to predict the face (int val)
#map the pred to a person/user
#display predicted name and bounding box around the face

import cv2
import numpy as np 
import os 

#KNN ALGORITHM
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# KNN ALGO
def knn(train,queryPoint,k=5):
    
    vals = []
    m = train.shape[0]
    
    for i in range(m):

    	x=train[i,:-1]	#vector 
    	y=train[i,-1]	#label
    	d = dist(queryPoint,x)
    	vals.append((d,y))
        
    #sort and get top k closest points
    vals = sorted(vals,key=lambda x:x[0])[:k]

    #retrieve labels
    labels=np.array(vals)[:,-1]
    
    vals = np.array(vals)

    #get frequency of each label
    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    #find the maximum frequency and corresponding label
    index = np.argmax(new_vals[1])
    #prediction
    pred = new_vals[0][index]
    
    return pred


#initialize webcam
cap=cv2.VideoCapture(0) #select which cam to use for capturing 

#face dedection 
mcascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#data preparation
class_id=0 #label for a given file
names={}#mapping id with name
datapath="./data/"
faceData=[]
labels=[]

for file in os.listdir(datapath):
	if file.endswith('.npy'):

		#establish mapping between class_id and name
		names[class_id]=file[:-4]

		dataitem=np.load(datapath+file)
		faceData.append(dataitem)

		#create labels for the class
		target=class_id*np.ones((dataitem.shape[0],))

		class_id+=1
		labels.append(target)

faceDataset=np.concatenate(faceData,axis=0)
faceLabels=np.concatenate(labels,axis=0).reshape((-1,1))

trainigSet=np.concatenate((faceDataset,faceLabels),axis=1)
#print(trainigSet.shape)

#testing part
while True:

	#Step1

 	check,frame = cap.read()

 	if check==False:	# ie, if the frame has not been captured properly
 		continue
 	#convert frame to gray
 	grayScale= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

 	facesDetected=mcascade.detectMultiScale(grayScale,1.3,5)
 	if len(facesDetected)==0:
 		continue

 	#Step2
 	
 	for face in facesDetected:
 		#draw the bounding box
 		x,y,w,h=face

 		#extract the area of interest: ie the face region
 		offset=10
 		faceRegion=grayScale[y-offset:y+h+offset,x-offset:x+w+offset]
 		faceRegion=cv2.resize(faceRegion,(100,100))

 		#predict 
 		output=knn(trainigSet,faceRegion.flatten())

 		#display the output 
 		predictedName=names[int(output)]
 		cv2.putText(grayScale,predictedName,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
 		cv2.rectangle(grayScale,(x,y),(x+w,y+h),(0,255,255),2)


 	cv2.imshow(" GraY Scale",grayScale)
 	
 	#cv2.imshow("Frame",frame)

 	keyPressed=cv2.waitKey(1) & 0xFF
 	if keyPressed== ord('q'):
 		break

cap.release()
cv2.destroyAllWindows()