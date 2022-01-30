from conut import MechanicalGrapheneBulk


class Plot:
    def __init__(self, system) -> None:
        if type(system) == MechanicalGrapheneBulk:
            pass


class PlotMechanicalGrapheneBulk(Plot):
    def __init__(self) -> None:
        super(PlotMechanicalGrapheneBulk, self).__init__()

    def dispersion3d(self):
        pass
