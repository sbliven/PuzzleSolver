import kivy
kivy.require('1.5.1')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

Builder.load_string('''
<MyWidget>:
    anchor_x: 'center'
    anchor_y: 'center'
    RelativeLayout:
        id: camlayout
        size: cam.size
        size_hint: None,None
        anchor_x: 'center'
        anchor_y: 'center'
        Camera:
            id: cam
            resolution: 640,480
            size: self.resolution
        Button:
            text: "buttz"
            size: 200,200
            size_hint: None,None
        Widget:
            size: cam.size
            size_hint: None,None
            canvas:
                Color:
                    rgba: 1,0,0,.5
                Rectangle:
                    pos: self.pos
                    size: self.size

''')

class MyWidget(BoxLayout):
    pass
class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == "__main__":
    MyApp().run()
