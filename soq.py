    import kivy
    kivy.require('1.5.1')

    from kivy.app import App
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.gridlayout import GridLayout
    from kivy.lang import Builder

    Builder.load_string('''
    <MyWidget>:
        cols:3
        Button:
            text:'A'
        Button:
            text:'B'
        Label:
            text:'text'
        Label:
            text:'other'
        Button:
            text:'text'
    ''')

    class MyWidget(GridLayout):
        pass
    class MyApp(App):
        def build(self):
            return MyWidget()

    if __name__ == "__main__":
        MyApp().run()
