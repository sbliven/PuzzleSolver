#!/usr/bin/python
"""
@author Spencer Bliven <sbliven@ucsd.edu>
"""

import sys
import os
import optparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import kivy
kivy.require('1.4.0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, AliasProperty
from kivy.vector import Vector
from kivy.clock import Clock

def texture2img(texture):
    """Extracts the pixels from a texture and put them into a numpy BGRA image"""
    img = np.fromstring(texture.pixels,np.uint8).reshape([texture.height,texture.width,4])
    return img

def img2texture(img,texture):
    """Takes the numpy BGRA image and replaces the texture data with it"""
    if img.shape != (texture.height,texture.width,4):
        raise ValueError("Image sizes don't match. Texture %s but image %s" %( (texture.height,texture.width,4),img.shape) )

    #pix = img.tostring()
    #texture.pixels = pix
    buf=img.flatten()
    texture.blit_buffer(buf,colorfmt='bgra',bufferfmt='ubyte')
    return texture


class Handle(Widget):
    center_x = AliasProperty(get_center_x,set_center_x, bind=('x','width'))
    center_y = AliasProperty(get_center_y,set_center_y, bind=('y','width'))
    center = ReferenceListProperty(center_x,center_y)

    def get_center_x(self):
        return self.x+self.width/2
    def set_center_x(self,val):
        self.x = val - self.width/2

    def get_center_y(self):
        return self.y+self.width/2
    def set_center_y(self,val):
        self.y = val - self.width/2

class Overlay(Widget):
    rows = NumericProperty(9)
    cols = NumericProperty(9)

    topleft     = ObjectProperty(None)
    topright    = ObjectProperty(None)
    bottomright = ObjectProperty(None)
    bottomleft  = ObjectProperty(None)

    def updateCorners(self,bgraimg):
        """Find the corners of the biggest contour, which should be the grid
        rgbaimg: a numpy greyscale image

        returns: the corners of the grid as a numpy array with rows topleft,
            topright, bottomright, bottomleft
        """
        img = cv2.cvtColor(bgraimg, cv2.COLOR_BGRA2GRAY)
        blurred = cv2.GaussianBlur(img,(11,11),0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
        #dilated = cv2.dilate(~thresholded, kernel, iterations=1)
        contourImg, contours, hierarchy = cv2.findContours((~thresholded).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        biggest = None
        max_area = 0
        for i in contours:
                area = cv2.contourArea(i)
                if area > 100:
                        peri = cv2.arcLength(i,True)
                        approx = cv2.approxPolyDP(i,0.02*peri,True)
                        if area > max_area and len(approx)==4:
                                biggest = approx
                                max_area = area
        tl,_,_,br = zip(*sorted(zip(np.sum(biggest.squeeze(),axis=1),biggest.squeeze())))[1]
        bl,_,_,tr = zip(*sorted(zip(biggest[:,0,0]-biggest[:,0,1],biggest.squeeze())))[1]

        topleft.center = list(tl)
        topright.center = list(tr)
        bottomright.center = list(br)
        bottomleft.center = list(bl)

        #self.corners = np.vstack([tl,tr,br,bl])
        #return self.corners

    def drawGrid(self,bgraimg):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(0,0,1,1)
        color = (0,0,255,255)
        thickness = 1

        if self.topleft is None:
            self.updateCorners(bgraimg)

        for row in xrange(self.height+1):
            alpha = float(row)/(self.height)
            # interpolate topleft to bottomleft
            left = (1-alpha)*self.corners[0,:]+alpha*self.corners[3,:]
            left = np.array(left,np.uint32)
            # interpolate topright to bottomright
            right = (1-alpha)*self.corners[1,:]+alpha*self.corners[2,:]
            right = np.array(right,np.uint32)
            cv2.line(bgraimg,tuple(left),tuple(right),color,thickness, cv2.LINE_AA)

        for row in xrange(self.width+1):
            alpha = float(row)/(self.width)
            top = (1-alpha)*self.corners[0,:]+alpha*self.corners[1,:]
            top = np.array(top,np.uint32)
            bottom = (1-alpha)*self.corners[3,:]+alpha*self.corners[2,:]
            bottom = np.array(bottom,np.uint32)
            cv2.line(bgraimg,tuple(top),tuple(bottom),color,thickness, cv2.LINE_AA)


class CameraWindow(BoxLayout):
    camera = ObjectProperty(None)

    def __init__(self,*args,**kwargs):
        super(CameraWindow,self).__init__(*args,**kwargs)

        self.grid = GridDetector(9,9)

        self.camera.bind(texture=self._on_texture)

    def _on_texture(self,camera,texture,*args, **kwargs):
        print(("_on_texture",camera,texture,args, kwargs))
        texture.add_reload_observer(self._on_reload)

    def _on_reload(self,context,*args, **kwargs):
        print(("_on_reload", context,args, kwargs))

    def capture(self):
        print 'Click'
        self.camera.play = False
        t = self.camera.texture
        img = texture2img(t)

        # Replace part of image with known sudoku board for testing
        fake = cv2.imread('sudoku-original.jpg',cv2.IMREAD_COLOR)
        fake = cv2.cvtColor(fake,cv2.COLOR_RGB2BGRA)
        img[:fake.shape[0],:fake.shape[1],:] = fake

        self.grid.updateCorners(img)
        self.grid.drawGrid(img)

        img2texture(img,t)
        pass



class PuzzleGrabberApp(App):

    def build(self):
        window = CameraWindow()
        return window

def captureFromCamera():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print "no frames"
            break
        imshow(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    video_capture.release()







if __name__ == "__main__":
    PuzzleGrabberApp().run()
