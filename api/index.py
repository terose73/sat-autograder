from http.server import BaseHTTPRequestHandler
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
        image = cv2.imread('omr_test_01.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cv2.imshow('img',edged)
        cv2.waitKey(0)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        docCnt = None
        if len(cnts) > 0:
            # Sorting contours based on their area. Paper will have mamimum area as it is the main object
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  
            
            for c in cnts:
                
                peri = cv2.arcLength(c, True)   # Calculating the perimeter
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)    # approximating the shape
                
                if len(approx) == 4:    # If shape has four edges
                    docCnt = approx
                    break
        cv2.imshow('temp',cv2.drawContours(image.copy(),[docCnt],-1,(0,0,255),2))
        cv2.waitKey(0)
        paper = four_point_transform(image,docCnt.reshape(4,2))
        warped = four_point_transform(gray,docCnt.reshape(4,2))
        # OTSU will automatically determine the best value for the parameter thresh. The value of thresh is not considered 
        # when otsu flag is passed
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]   
        cv2.waitKey(0)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)     # Computing the bounding box for the bubble 
            ar = w / float(h)     # Computing the aspect ratio of the bounding box
            
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        pap = paper.copy()
        q = -1
        correct = 0
        flag = False
        minm = []
        maxm = []
        for i in range(0,len(questionCnts),5):
            q+=1
            
            # Sorting contours from left to right
            temp = contours.sort_contours(questionCnts[i:i+5], method="left-to-right")[0]
            
            bubbled = None
            flag = False
            for (j,c) in enumerate(temp):
                
                mask = np.zeros(thresh.shape, dtype="uint8")     # Create a dummy mask
                cv2.drawContours(mask, [c], -1, 255, -1)         # Drawing the contours on the dummy mask
                                                                            
                
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)     # Doing Bitwise-and to reveal the bubble
                
                total = cv2.countNonZero(mask)     # Calculating the sum of non-zero pixels            
                
                # Breaking the loop and setting bubbled = None when another marked bubble is found in same question
                if flag and total > 600:     
                    bubbled = None
                    break
                    
                if total > 600:     # Setting flag = True when a marked bubble is found
                    flag = True
                    
                if total > 600:     # Comparing the sum of non zero pixels
                    bubbled = (total, j)
                        
            color = (0, 0, 255)
            k = ANSWER_KEY[q]     # Retrieving the answer from the answer key based on question 'q'
            
            if bubbled and k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1
            if bubbled:     # Not Drawing contour when bubbled is None
                cv2.drawContours(pap, [temp[k]], -1, color, 3)
        score = (correct /(q+1)) * 100
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        message = str(score)
        self.wfile.write(message.encode())