import math
import cv2
import mediapipe as mp
import time
import numpy as np


class BallonGame:
    def __init__(self):
        self.targets = []
        self.score = 0
        self.gravity = 2
        self.popped_ballon = 0
        self.target_spawn_interval = 1  # in seconds
        self.last_spawn_time = time.time()
        self.cap_width = 900
        self.cap_height = 1200

        #set up video capture
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow("Reaction Game", cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                                        max_num_hands=2,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

        self.pTime = time.time()

    def on_hand_closed(self):
        pass

    def on_hand_open(self):
        pass

    def spawn_target(self):
        if time.time() - self.last_spawn_time > self.target_spawn_interval:
            self.last_spawn_time = time.time()
            target_position = [np.random.randint(0,1500),0]
            self.targets.append({'position': target_position, 'radius':20,'velocity': [0, 0],})

        self.targets = [target for target in self.targets if target['position'][1] < self.cap_height + 300]
        for target in self.targets:
            target['position'][0] += target['velocity'][0]
            target['position'][1] += target['velocity'][1]
            target['velocity'][1] += self.gravity

    

    
        

    def on_key_pressed(self,key):
        if key == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            '''
            start game
            '''
            pass
        elif key ==('f'):
            '''
            full screen functionality
            '''
            pass


    def run_game(self):
        while True:

            ret, frame = self.cap.read()

            if not ret:
                break
            
            frame = cv2.flip(frame,1)   #flips camera position
            imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x,lm.y] for lm in handLms.landmark])

                index_tip = landmarks[8]
                self.index_cx, self.index_cy = int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])

                is_hand_closed = landmarks[5,1] < landmarks[8,1] #checks if tip of index is lower than base of index

                #pause game if hand is closed
                if is_hand_closed:
                    cv2.putText(frame, "Paused", (450, 600), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 6)


                else:
                    self.spawn_target()
                    self.targets = [target for target in self.targets if
                    np.linalg.norm(np.array(target['position']) - np.array([self.index_cx, self.index_cy])) >= 90]

                    for target in self.targets:
                        if target['position'][1] > self.cap_height + 286:
                            self.score += -5

                        elif np.linalg.norm(np.array(target['position']) - np.array([self.index_cx, self.index_cy])) < 90:
                            self.score += 10
                            


                    
                    print(self.targets)



                for target in self.targets:
                    cv2.circle(frame, (int(target['position'][0]), int(target['position'][1])), target['radius'], (0, 0, 255), cv2.FILLED)
        
                self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS) # shwos the mask of the hand

                
            


            cTime = time.time()
            fps = 1/ (cTime - self.pTime)
            self.pTime = cTime
            

            cv2.putText(frame, f"Score: {(self.score)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Reaction Game", frame)

            key = cv2.waitKey(1)
            self.on_key_pressed(key)

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

# Create an instance of the BallonGame class and run the game
reaction_game = BallonGame()
reaction_game.run_game() 

