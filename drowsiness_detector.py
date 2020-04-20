# Imports for Gui
from tkinter import *
from tkinter import messagebox

# Import necessary libraries for Drowsiness detection
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2


# root window initialization
root = Tk()

# Drowsiness detection section
#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')


#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Has parameters that corresponds to different speeds enabling it to check diff conditions.
def drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES):
    #Start webcam video capture
    video_capture = cv2.VideoCapture(0)

    #Give some time for camera to initialize(not required)
    time.sleep(1)

    while True:
        #Read each frame and flip it, and convert to grayscale
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detect facial points through detector function
        faces = detector(gray, 0)

        #Detect faces through haarcascade_frontalface_default.xml
        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw rectangle around each face detected
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Detect facial points
        for face in faces:

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            #Get array of coordinates of leftEye and rightEye
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            #Calculate aspect ratio of both eyes
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            #Use hull to remove convex contour discrepencies and draw eye shape around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            #Detect if eye aspect ratio is less than threshold
            if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
                COUNTER += 1
                #If no. of frames is greater than threshold frames,
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            else:
                pygame.mixer.music.stop()
                COUNTER = 0

        #Show video feed
        cv2.imshow('Video', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    #Finally when video capture is over, release the video capture and destroyAllWindows
    video_capture.release()
    cv2.destroyAllWindows()


#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear


#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# This function gets speed from the user, starts video capture and drowsiness detection(drowsiness_checker)
def get_speed():
    try:
        # input speed as an interger
        speed = int(speed_input.get())
        # checks speed range
        if speed == 0:
            messagebox.showwarning('Invalid Speed','Vehicle is stationary at this speed')
            speed_input.delete(0, END)            
        elif speed > 600:
            messagebox.showwarning('Invalid Speed','Speed entered is out of range')
            speed_input.delete(0, END)
        elif speed < 0:
            messagebox.showwarning('Invalid Speed','Speed entered is out of range')
            speed_input.delete(0, END)  
#   This part contains most of the logic
        else:
            # This is where the video_capture code should go into-Speed is within range
            # Limits are proposed threshold for the fatigue detection system
            if speed > 0 and speed < 50:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 22

                #Counts no. of consecutive frames below threshold value
                COUNTER = 0       
                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
            
            elif speed >= 50 and speed < 60:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 20

                #Counts no. of consecutive frames below threshold value
                COUNTER = 0       

                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)

            elif speed >= 60 and speed < 70:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 16

                #COunts no. of consecutive frames below threshold value
                COUNTER = 0       

                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
          
            elif speed >= 70 and speed < 80:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 13

                #COunts no. of consecutive frames below threshold value
                COUNTER = 0       

                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
                  
            elif speed >= 80 and speed < 90:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 11

                #COunts no. of consecutive frames below threshold value
                COUNTER = 0       

                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
                  
            elif speed >= 90 and speed < 100:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 9

                #Counts no. of consecutive frames below threshold value
                COUNTER = 0       

                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
                  
            elif speed >= 100:
                #Minimum threshold of eye aspect ratio below which alarm is triggered
                EYE_ASPECT_RATIO_THRESHOLD = 0.3

                #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
                EYE_ASPECT_RATIO_CONSEC_FRAMES = 6

                #COunts no. of consecutive frames below threshold value
                COUNTER = 0       

                # Use above threshold in drowsiness checker
                drowsiness_checker(COUNTER,EYE_ASPECT_RATIO_THRESHOLD,EYE_ASPECT_RATIO_CONSEC_FRAMES)
   
    # If input is not an integer
    except ValueError:
        messagebox.showerror('Value Error','Please enter a valid number')
        speed_input.delete(0, END)
    except:
        messagebox.showerror('Error','Exiting program, Please launch again')
        root.quit()


# stops and ends program just like quit
def stop_program():
    messagebox.showwarning('Program Stopped','Thank you for using the program')
    root.quit()


# title and resolution
root.title("Drowsiness Detector Program")
root.geometry('500x200')

# GUIDELINES 
guide_label = Label(root, text='Please Enter Vehicle Speed(km/h) & Press Submit to Launch the Program')
guide_label.grid(row=0,column=0,columnspan=3)

# speed entry & label
speed_label = Label(root,text='Enter Vehicle speed(km/h):', height=2)
speed_label.grid(row=1,column=0)

speed=StringVar() 
speed_input = Entry(root, textvariable=speed)
speed_input.grid(row=1, column=1)

speed_button = Button(root,text='Submit', command=get_speed,width=15)
speed_button.grid(row=1,column=2)

# clear entry label
clear_entry_label = Label(root,text='Clear/Refresh Entries', width=20,height=2)
clear_entry_label.grid(row=2,column=0)

clear_entry_button = Button(root,text='Clear',width=15,command=lambda:speed_input.delete(0, END))
clear_entry_button.grid(row=2,column=1)


# stop label and button
stop_label = Label(root,text='Stop The program', width=20,height=2)
stop_label.grid(row=3,column=0)

stop_button = Button(root, text='Stop',width=15,command=stop_program)
stop_button.grid(row=3,column=1)

# End Prog button and label
end_program = Label(root,text='End The Program', width=20,height=2)
end_program.grid(row=4,column=0)

end_button = Button(root,text='Quit', command=quit, width=15)
end_button.grid(row=4,column=1)

# maximum window size
root.maxsize(500,200)
root.mainloop()
