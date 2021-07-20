# Capture images from videa stream 
# Detect faces and show bounding box (haarcascade)
# 	haarcascade is a pretrained classifier on cnn 
# 	detect faces and draw the bounding box around this face using the haarcascade classifier
# Flatten the face image and store it as a numpy array



import cv2 #computer vision module
import numpy as np

#initialize webcam
cap=cv2.VideoCapture(0) #select which cam to use for capturing 

#face dedection 
mcascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#face data
faceData=[]
datapath= "./data/"
filename=input('Enter the name of the person whose datasets needs to be created\n')
#reading the frames 
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

 	facesDetected=sorted(facesDetected,key=lambda f:f[2]*f[3])

 	#Step2
 	
 	for face in facesDetected[-1:]:
 		#draw the bounding box
 		x,y,w,h=face
 		cv2.rectangle(grayScale,(x,y),(x+w,y+h),(0,255,255),2)

 		#extract the area of interest: ie the face region
 		offset=10
 		faceRegion=grayScale[y-offset:y+h+offset,x-offset:x+w+offset]
 		faceRegion=cv2.resize(faceRegion,(100,100))
 		faceData.append(faceRegion)
 		print(len(faceRegion))

 	cv2.imshow(" GraY Scale",grayScale)
 	
 	#cv2.imshow("Frame",frame)

 	keyPressed=cv2.waitKey(1) & 0xFF
 	if keyPressed== ord('q'):
 		break


#Step3 
#convert face data into numpy array
faceData=np.asarray(faceData)
faceData=faceData.reshape((faceData.shape[0],-1))

print("\n",faceData.shape)
print(faceData.shape[0],"frames captured")

#save data into a file
np.save(datapath+filename+'.npy',faceData) 
print("Successfully saved dataset")

cap.release()
cv2.destroyAllWindows()