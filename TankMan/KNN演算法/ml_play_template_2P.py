"""
The template of the main script of the machine learning process
"""
#python -m mlgame -i ./ml/ml_play_template2.py . --difficulty NORMAL --level 5 
#使用knn
import pygame
from sklearn import neighbors
import os
import sys
import io
import pickle
import csv
import random
import time
# Describe this function...
class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.model = None
        self.side = ai_name
        self.time = 0
        DIR = os.path.dirname(__file__)
        #load history data
        #把model.pockle 儲存到 self.model
        if os.path.exists(DIR+'/model2.pickle'):
            with open(DIR+'/model2.pickle', 'rb') as f:
                self.model = pickle.load(f)
            print('model loaded')
        self.bullet_stations_info = False
        self.record2 = [] #frame,ballX,ballY,dx,dy,platformX
        self.features = []
        self.targets = []
        self.offset = 600#**
        self.mapSize = int((855-self.offset)/10*40)#int((390-self.offset)/10*40)**
        print("=============================")

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        # print(scene_info)
        # print(keyboard)
        command = None
        if keyboard is None:keyboard = []
        if scene_info["status"] == "GAME_AIIVE":
            command = "RESET"
        else :
            
            #command = "TURN_LEFT"
            frame = scene_info['used_frame']
            x = scene_info['competitor_info']['x']
            y = scene_info['competitor_info']['y']
            platformy = scene_info['y']
            dx,dy = 0,0
            if frame>0 and len(self.record2)>0:
                lastFrame,lastX,lastY,last_dx,last_dy = self.record2[-1][0:5]
                dx,dy = last_dx,last_dy
                if frame - lastFrame==1:
                    dx,dy = (x-lastX)*50,(y-lastY)*50
                    #if dx!=last_dx or dy!=last_dy:print('dx,dy=',dx,dy)
            command = "NONE"
            #print(x)
            fv = [frame,x,y,dx,dy]
            self.record2.append(fv)
            self.side = "2P"
            #command = "SHOOT"
            #time.sleep(0.5)
            if self.side == "2P":
                #command = "SHOOT"
                #z = scene_info['angle']
                #print(z)
                #time.sleep(0.5)
                #print("ok")
                #print(fv)
                #print(platformy)
                if self.model!=None and x>0:#platform length = 40
                    #print(z)
                    targety = self.model.predict([fv[1:]])[0]
                    #print("ok")
                    time.sleep(0.3)
                    if platformy > targety:  command = "SHOOT"
                    elif platformy < targety:command = "SHOOT"
                
            if pygame.K_UP in keyboard:  command = "FORWARD"
            elif pygame.K_DOWN in keyboard: command = "BACKWARD"
        return command

    def reset(self):
        DIR = os.path.dirname(__file__)
        #load history data
        if os.path.exists(DIR+'/targets2.pickle'):
            with open(DIR+'/targets2.pickle', 'rb') as f:
                self.targets = pickle.load(f)
            with open(DIR+'/features2.pickle', 'rb') as f:
                self.features = pickle.load(f)
        
        #save record
        with open(DIR+ '/record2' +'.csv', 'w', newline='') as f:
            csv.writer(f, delimiter=',').writerows(self.record2)
        #model training
        self.record2.reverse()
        qq = None

        for i in range(len(self.record2)):
            #print("ok")
            x=self.record2[i][1]
            #print(i)
        targety = self.record2[i][2] #x
        Temp = []
        i=0
        for fv in self.record2[qq:]:
            frame,x,y,dx,dy = fv[0:5]
            self.features.append(fv[1:])
            self.targets.append([targety])
            temp=[targety]
            temp.extend(fv)
            Temp.append(temp)
            i+=1
            
            if x==0 and i>4:break    
        print("增加feature2數量:",i)
        self.model = neighbors.KNeighborsRegressor(7, weights='distance')
        self.model.fit(self.features,self.targets)
        #save features <==> targets
        with open(DIR+ '/features2' +'.csv', 'w', newline='') as f:
            csv.writer(f, delimiter=',').writerows(Temp)
        with open(DIR+ '/features2' + '.pickle', 'wb') as f:
            pickle.dump(self.features, f)
        with open(DIR+ '/targets2' + '.pickle', 'wb') as f:
            pickle.dump(self.targets, f)
        with open(DIR+ '/model2' + '.pickle', 'wb') as f:
            pickle.dump(self.model, f)
        #reset parameter
        self.bullet_stations_info = False
        self.record2 = []
        
