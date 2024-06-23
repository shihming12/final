import pygame
from sklearn import neighbors
import os
import sys
import io
import pickle
import csv
import random

#python -m mlgame -i ./ml/ml_play_template_1P.py -i ./ml/ml_play_template_2P.py  ./ --difficulty HARD --game_over_score 3  --init_vel 10

class MLPlay:
    def __init__(self,  ai_name,*args,**kwargs):
        """
        Constructor

        @param ai_name A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.model = None
        DIR = os.path.dirname(__file__)
        #load history data
        #把model.pockle 儲存到 self.model
        if os.path.exists(DIR+'/model2.pickle'):
            with open(DIR+'/model2.pickle', 'rb') as f:
                self.model = pickle.load(f)
            print('model1 loaded2')
        self.ball_served = False
        self.record2 = [] #frame,ballX,ballY,dx,dy,platformX
        self.features = []
        self.targets = []
        self.offset = 200
        self.mapSize = int((340-self.offset)/10*40)#int((390-self.offset)/10*40)
        print("=============================")

    def update(self, scene_info, keyboard=None, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        command = None
        if keyboard is None:keyboard = []

        if not self.ball_served:
            command = "SERVE_TO_RIGHT"
            self.ball_served = True

        else:
            frame = scene_info['frame']
            x,y = scene_info['ball']
            platformX = scene_info['platform_2P'][0]
            dx,dy = 0,0
            if frame>0 and len(self.record2)>0:
                lastFrame,lastX,lastY,last_dx,last_dy = self.record2[-1][0:5]
                dx,dy = last_dx,last_dy
            
                if frame - lastFrame==1:
                    dx,dy = (x-lastX)*5,(y-lastY)*5
                    #if dx!=last_dx or dy!=last_dy:print('dx,dy=',dx,dy)
            
            command = "MOVE_LEFT"
            fv = [frame,x,y,dx,dy]
            self.record2.append(fv)
            
            if self.model!=None and y>80+7:#platform length = 40
                targetX = self.model.predict([fv[1:]])[0] #除了 fv 的第一個除外 frame "x" "y" "dx" "dy" "Map" 返回一個值給targetX 也就是預測值
                if platformX+20 > targetX:  command = "MOVE_LEFT"
                elif platformX+20 <= targetX:command = "MOVE_RIGHT"


            if y==70: print('hit at frame2 ',frame)           
            if pygame.K_LEFT in keyboard:  command = "MOVE_LEFT"
            elif pygame.K_RIGHT in keyboard: command = "MOVE_RIGHT"
        return command

    def reset(self):
        """
        Reset the status
        """
        print("=============================")
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
        qq = 0
        for i in range(len(self.record2)):
            y=self.record2[i][2]
            if y>=(80-7) and y<=(80+7):qq = i;break
        targetX = self.record2[qq][1] #x
        Temp = []
        i=0
        for fv in self.record2[qq:]:
            frame,x,y,dx,dy = fv[0:5]
            self.features.append(fv[1:])
            self.targets.append([targetX])
            temp=[targetX]
            temp.extend(fv)
            Temp.append(temp)
            i+=1
            if y==80 and i>4:break
            
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
        self.ball_served = False
        self.record2 = []
        