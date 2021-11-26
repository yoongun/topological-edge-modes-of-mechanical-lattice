from kivy.app import App
from kivy.uix.widget import Widget
from meclat import MechanicalLattice1D


class MyPaintWidget(Widget):
    pass


class MyPaintApp(App):
    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    ml1d = MechanicalLattice1D(k=[1., 1.], m=[1.3, .7])
    dispersion, eigenvecs = ml1d.dispersion()
    phonon_mode = dispersion[:, 0]
    photon_mode = dispersion[:, 1]
    MyPaintApp().run()
