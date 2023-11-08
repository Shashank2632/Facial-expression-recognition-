import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
from deepface import DeepFace

r = sr.Recognizer()

machine = pyttsx3.init()

def talk(text):
     machine.say(text)
     machine.runAndWait()

def get_instruction():
    try:
       with sr.Microphone(device_index=0) as source:
          print("Listening...")
          r.adjust_for_ambient_noise(source)
          speech = r.listen(source)
          instruction = r.recognize_google(speech)
          instruction = instruction.lower()
          if "hi" in instruction:
             instruction = instruction.replace('hi', "")
             print(instruction)
             return instruction
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return None

def expression():
    # haar_cascade =cv.CascadeClassifier('haar.xml')
    cap=cv.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        result = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)

        gray =cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        # faces_rect =haar_cascade.detectMultiScale(gray,1.2,4)

        # for (x,y,w,h) in faces_rect:
        #     cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)
        talk('Dominant emotion is'+result[0]['dominant_emotion'])
        break
        # cv.putText(frame,result[0]['dominant_emotion'],(200,100),cv.FONT_HERSHEY_COMPLEX,3,(0,0,255),7,cv.LINE_4)    
        # cv.imshow('Original Video',frame)
        
        # if cv.waitKey(2) & 0xFF == ord('d'):
        #     break
    cap.release()
    cv.destroyAllWindows()



def play_instruction():
    instruction = get_instruction()
    print(instruction)    
    if instruction is not None and "play" in instruction:
       song = instruction.replace('play', "") 
       talk("Playing " + song)   
       pywhatkit.playonyt(song)
   
    elif instruction is not None and 'time' in instruction:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('The current time is ' + time)
    elif instruction is not None and 'expression' in instruction:
        talk("Opening expression detection")
        expression()

play_instruction()
