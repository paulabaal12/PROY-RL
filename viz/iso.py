"""Proyección isométrica y ubicación de las estaciones en la cocina."""

# Cocina de 5 x 5 baldosas. Fila del fondo (y = 0): preparación y los 3 fuegos.
# Columna izquierda (x = 0): basurero y ventana de entrega. El resto son mostradores.
FLOOR = 5
STATION_TILES = {
    "prep": (1, 0),
    "burner0": (2, 0),
    "burner1": (3, 0),
    "burner2": (4, 0),
    "trash": (0, 2),
    "serve": (0, 4),
    "counter": [(0, 0), (0, 1), (0, 3)],
}
# Punto del piso donde se para el chef para usar cada estación
STAND_TILES = {
    "prep": (1.5, 1.35),
    "burner0": (2.5, 1.35),
    "burner1": (3.5, 1.35),
    "burner2": (4.5, 1.35),
    "trash": (1.35, 2.5),
    "serve": (1.35, 4.5),
    "center": (2.6, 2.7),
}
COUNTER_HEIGHT = 18


class Iso:
    def __init__(self, origin: tuple[float, float], tile_w: float, tile_h: float):
        self.ox, self.oy = origin
        self.tile_w = tile_w
        self.tile_h = tile_h

    @property
    def k(self) -> float:
        """Escala respecto al mockup (baldosa de 50 px)."""
        return self.tile_w / 50

    def to_screen(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        return (self.ox + (x - y) * self.tile_w / 2,
                self.oy + (x + y) * self.tile_h / 2 - z * self.k)
