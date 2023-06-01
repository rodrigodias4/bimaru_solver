# Grupo 135:
# 99552 Rodrigo Dias

import sys
from sys import stdin, stdout
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    depth_first_graph_search,
    greedy_search,
    recursive_best_first_search,
)

Action = (int, int, int, int)

HORIZONTAL = 1
VERTICAL = 0
D_NONE = 0
D_MINIMUM = 1
D_VERBOSE = 2
DEBUG_LEVEL = D_NONE
A_ROW = 0
A_COL = 1
A_SIZE = 2
A_DIR = 3


def hv(x: int) -> str:
    if x:
        return "H"
    return "V"


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(
        self,
        grid: np.ndarray,
        ships: np.ndarray,
        count_row: np.ndarray,
        count_column: np.ndarray,
    ):
        self.grid = grid
        self.count_row = count_row
        self.count_column = count_column
        self.ships = ships
        self.water_fill()

    def get_value(self, row: int, col: int) -> str:
        if not Board.within_bounds(row, col):
            return ""
        return self.grid[row, col]

    def is_water(self, r, c) -> bool:
        if not Board.within_bounds(r, c):
            return False
        v = self.get_value(r, c)
        return v in {".", "W"}

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        above = "None"
        below = "None"
        if Board.within_bounds(row - 1, col):
            if self.get_value(row - 1, col) != "":
                above = self.get_value(row - 1, col)
        if Board.within_bounds(row + 1, col):
            if self.get_value(row + 1, col) != "":
                below = self.get_value(row + 1, col)
        return (above, below)

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left = "None"
        right = "None"
        if Board.within_bounds(row, col - 1):
            if self.get_value(row, col - 1) != "":
                left = self.get_value(row, col - 1)
        if Board.within_bounds(row, col + 1):
            if self.get_value(row, col + 1) != "":
                right = self.get_value(row, col + 1)
        return (left, right)

    def is_water_or_oob(self, r: int, c: int):
        """Verifica se a célula é água ou está fora dos limites"""
        if not Board.within_bounds(r, c):
            return True
        else:
            if self.is_water(r, c):
                return True
            else:
                return False

    def water_if_empty(self, r: int, c: int):
        """Coloca água na célula se esta estiver vazia"""
        if Board.within_bounds(r, c):
            if self.grid[r, c] == "":
                self.grid[r, c] = "."

    def set_if_empty_and_within_bounds(self, r: int, c: int, value: str):
        """Coloca o valor indicado na célula se esta estiver vazia"""
        if Board.within_bounds(r, c):
            if self.get_value(r, c) == "":
                self.grid[r, c] = value

    def inferences(self):
        for r in range(10):
            if self.ships_in_row(r) == self.count_row[r]:
                for i in range(10):
                    self.water_if_empty(r, i)
            if self.ships_in_column(r) == self.count_column[r]:
                for i in range(10):
                    self.water_if_empty(i, r)

            for c in range(10):
                if self.get_value(r, c).lower() == "m":
                    if r == 0:
                        self.water_if_empty(r + 1, c)
                    elif r == 9:
                        self.water_if_empty(r - 1, c)

                    if c == 0:
                        self.water_if_empty(r, c + 1)
                    elif c == 9:
                        self.water_if_empty(r, c - 1)

                    if self.ships_in_row(r) + 2 > self.count_row[r]:
                        self.water_if_empty(r, c - 1)
                        self.water_if_empty(r, c + 1)
                    elif self.ships_in_column(c) + 2 > self.count_column[c]:
                        self.water_if_empty(r - 1, c)
                        self.water_if_empty(r + 1, c)

                elif self.get_value(r, c).lower() == "t":
                    if self.is_water_or_oob(r + 2, c):
                        # CONTRATORPEDEIRO (2 CELULAS)
                        self.set_if_empty_and_within_bounds(r + 1, c, "b")
                    elif self.get_value(r + 3, c).lower() == "m":
                        # CONTRATORPEDEIRO (2 CELULAS) e OUTRO NAVIO
                        self.water_if_empty(r + 2, c)
                        self.water_if_empty(r + 4, c)
                        self.set_if_empty_and_within_bounds(r + 1, c, "b")
                    elif self.get_value(r + 2, c).lower() == "b":
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r + 1, c, "m")
                    elif self.get_value(
                        r + 1, c
                    ).lower() == "m" and self.is_water_or_oob(r + 3, c):
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r + 2, c, "b")
                        self.water_bottom(r + 2, c)
                    elif self.get_value(
                        r + 2, c
                    ).lower() == "m" and self.is_water_or_oob(r + 4, c):
                        # COURAÇADO (4 CELULAS)
                        self.set_if_empty_and_within_bounds(r + 3, c, "b")
                        self.set_if_empty_and_within_bounds(r + 1, c, "m")
                        self.water_bottom(r + 3, c)

                elif self.get_value(r, c).lower() == "b":
                    if self.is_water_or_oob(r - 2, c):
                        # CONTRATORPEDEIRO (2 CELULAS)
                        self.set_if_empty_and_within_bounds(r - 1, c, "t")
                    elif self.get_value(r - 3, c).lower() == "m":
                        # CONTRATORPEDEIRO (2 CELULAS) e OUTRO NAVIO
                        self.water_if_empty(r - 2, c)
                        self.water_if_empty(r - 4, c)
                        self.set_if_empty_and_within_bounds(r - 1, c, "t")
                    elif self.get_value(r - 2, c).lower() == "t":
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r - 1, c, "m")
                    elif self.get_value(
                        r - 1, c
                    ).lower() == "m" and self.is_water_or_oob(r - 3, c):
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r - 2, c, "t")
                        self.water_top(r - 2, c)
                    elif self.get_value(
                        r - 2, c
                    ).lower() == "m" and self.is_water_or_oob(r - 4, c):
                        # COURAÇADO (4 CELULAS)
                        self.set_if_empty_and_within_bounds(r - 3, c, "t")
                        self.set_if_empty_and_within_bounds(r - 1, c, "m")
                        self.water_top(r - 3, c)

                elif self.get_value(r, c).lower() == "l":
                    if self.is_water_or_oob(r, c + 2):
                        # CONTRATORPEDEIRO (2 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c + 1, "r")
                    elif self.get_value(r, c + 3).lower() == "m":
                        # CONTRATORPEDEIRO (2 CELULAS) e OUTRO NAVIO
                        self.water_if_empty(r, c + 2)
                        self.water_if_empty(r, c + 4)
                        self.set_if_empty_and_within_bounds(r, c + 1, "r")
                    elif self.get_value(r, c + 2).lower() == "b":
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c + 1, "m")
                    elif self.get_value(
                        r, c + 1
                    ).lower() == "m" and self.is_water_or_oob(r, c + 3):
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c + 2, "r")
                        self.water_right(r, c + 2)
                    elif self.get_value(
                        r, c + 2
                    ).lower() == "m" and self.is_water_or_oob(r, c + 4):
                        # COURAÇADO (4 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c + 3, "r")
                        self.set_if_empty_and_within_bounds(r, c + 1, "m")
                        self.water_right(r, c + 3)

                elif self.get_value(r, c).lower() == "r":
                    if self.is_water_or_oob(r, c - 2):
                        # CONTRATORPEDEIRO (2 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c - 1, "l")
                    elif self.get_value(r, c - 3).lower() == "m":
                        # CONTRATORPEDEIRO (2 CELULAS) e OUTRO NAVIO
                        self.water_if_empty(r, c - 2)
                        self.water_if_empty(r, c - 4)
                        self.set_if_empty_and_within_bounds(r, c - 1, "l")
                    elif self.get_value(r, c - 2).lower() == "r":
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c - 1, "m")
                    elif self.get_value(
                        r, c - 1
                    ).lower() == "m" and self.is_water_or_oob(r, c - 3):
                        # CRUZADOR (3 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c - 2, "l")
                        self.water_left(r, c - 2)
                    elif self.get_value(
                        r, c - 2
                    ).lower() == "m" and self.is_water_or_oob(r, c - 4):
                        # COURAÇADO (4 CELULAS)
                        self.set_if_empty_and_within_bounds(r, c - 3, "l")
                        self.set_if_empty_and_within_bounds(r, c - 1, "m")
                        self.water_left(r, c - 3)

    @staticmethod
    def parse_instance():
        initial_grid = np.empty([10, 10], str)
        count_row = np.array(stdin.readline().rstrip("\n").split("\t")[1:], np.ubyte)
        count_column = np.array(stdin.readline().rstrip("\n").split("\t")[1:], np.ubyte)

        n_hints = int(stdin.readline().rstrip("\n"))

        for i in range(n_hints):
            line = stdin.readline().rstrip("\n").split("\t")[1:]
            initial_grid[int(line[0]), int(line[1])] = line[2]

        initial_board = Board(
            initial_grid, np.zeros(4, np.ubyte), count_row, count_column
        )

        while True:
            previous_board = Board(
                np.array(initial_board.grid, str),
                np.array(initial_board.ships, np.ubyte),
                np.array(initial_board.count_row, np.ubyte),
                np.array(initial_board.count_column, np.ubyte),
            )
            initial_board.inferences()
            if initial_board.__eq__(previous_board):
                break

        initial_board.ship_count()

        if not initial_board.valid_board():
            sys.exit("Tabuleiro inicial inválido")
        else:
            return initial_board

    @staticmethod
    def within_bounds(r: int, c: int):
        return r >= 0 and r < 10 and c >= 0 and c < 10

    def ship_count(self):
        self.ships = np.zeros(4, np.ubyte)

        for r in range(10):
            for c in range(10):
                if self.get_value(r, c).lower() == "c":
                    self.ships[0] += 1
                if self.get_value(r, c).lower() == "t":
                    for i in range(1, 4):
                        if self.get_value(r + i, c).lower() == "b":
                            self.ships[i] += 1
                        if self.get_value(r + i, c).lower() != "m":
                            break
                if self.get_value(r, c).lower() == "l":
                    for i in range(1, 4):
                        if self.get_value(r, c + i).lower() == "r":
                            self.ships[i] += 1
                        if self.get_value(r, c + i).lower() != "m":
                            break

    def pontas_soltas(self) -> bool:
        for r in range(10):
            for c in range(10):
                v = self.get_value(r, c).lower()
                if v == "t":
                    temp = self.get_value(r + 1, c).lower()
                    if temp not in {"m", "b"}:
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"t {r} {c}")
                        return False
                elif v == "b":
                    temp = self.get_value(r - 1, c).lower()
                    if temp not in {"m", "t"}:
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"b {r} {c} {temp}")
                        return False
                elif v == "l":
                    temp = self.get_value(r, c + 1).lower()
                    if temp not in {"m", "r"}:
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"l {r} {c}")
                        return False
                elif v == "r":
                    temp = self.get_value(r, c - 1).lower()
                    if temp not in {"m", "l"}:
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"r {r} {c}")
                        return False
                elif v == "m":
                    adjacent_to_m = (
                        int(self.is_water_or_oob(r - 1, c))
                        + int(self.is_water_or_oob(r + 1, c))
                        + int(self.is_water_or_oob(r, c - 1))
                        + int(self.is_water_or_oob(r, c + 1))
                    )
                    if adjacent_to_m != 2:
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"m {r} {c}")
                        return False
        for i in range(10):
            if self.ships_in_row(i) != self.count_row[i]:
                return False
            if self.ships_in_column(i) != self.count_column[i]:
                return False
        return True

    def valid_board(self) -> bool:
        for i in range(4):
            if self.ships[i] > 4 - i:
                return False

        for c in range(10):
            count = self.ships_in_column(c)
            if count > self.count_column[c]:
                if DEBUG_LEVEL >= D_VERBOSE:
                    print(
                        "valid_board: Column ",
                        c,
                        " com mais células com navios (",
                        count,
                        ") que as ajudas (",
                        self.count_column[c],
                        ")",
                    )
                return False

        for r in range(10):
            count = self.ships_in_row(r)
            if count > self.count_row[r]:
                if DEBUG_LEVEL >= D_VERBOSE:
                    print(
                        "valid_board: Row ",
                        r,
                        " com mais células com navios (",
                        count,
                        ") que as ajudas (",
                        self.count_row[r],
                        ")",
                    )
                return False

            for c in range(10):
                if self.ship_in_cell(r, c):
                    if (r, c) in {(0, 0), (0, 9), (9, 0), (9, 9)} and self.get_value(
                        r, c
                    ).lower() == "m":
                        # M no canto não é uma posição válida
                        if DEBUG_LEVEL >= D_VERBOSE:
                            print("valid_board: M no canto")
                        return False

                    if not self.check_diagonals(r, c):
                        # Diagonais
                        if DEBUG_LEVEL >= D_VERBOSE:
                            print("valid_board: Diagonais em ", r, c)
                            return False

                    v = self.get_value(r, c)
                    vl = v.lower()

                    if (
                        (vl == "t" and self.ship_in_cell(r - 1, c))
                        or (vl == "b" and self.ship_in_cell(r + 1, c))
                        or (vl == "l" and self.ship_in_cell(r, c - 1))
                        or (vl == "r" and self.ship_in_cell(r, c + 1))
                    ):
                        if DEBUG_LEVEL >= D_VERBOSE:
                            print("valid_board: Adjacente na ponta em ", r, c)
                        return False

                    if v in {"T", "B"}:
                        (left, right) = self.adjacent_horizontal_values(r, c)
                        if self.ship_in_cell_value(left) or self.ship_in_cell_value(
                            right
                        ):
                            if DEBUG_LEVEL >= D_VERBOSE:
                                print(
                                    "valid_board: Adjacente perpendicularmente em ",
                                    r,
                                    c,
                                )
                            return False
                    elif v in {"L", "R"}:
                        (above, below) = self.adjacent_vertical_values(r, c)
                        if self.ship_in_cell_value(above) or self.ship_in_cell_value(
                            below
                        ):
                            if DEBUG_LEVEL >= D_VERBOSE:
                                print(
                                    "valid_board: Adjacente perpendicularmente em ",
                                    r,
                                    c,
                                )
                            return False

                    if vl == "m":
                        if (
                            "m" in self.adjacent_horizontal_values(r, c)
                            or "M" in self.adjacent_horizontal_values(r, c)
                        ) and (
                            "m" in self.adjacent_vertical_values(r, c)
                            or "M" in self.adjacent_vertical_values(r, c)
                        ):
                            if DEBUG_LEVEL >= D_VERBOSE:
                                print("valid_board: Interseção perpendicular em ", r, c)
                            return False
        return True

    def water_bottom(self, r, c):
        """Figura:
        .       .
        .   B   .
        .   .   .
        """
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r, c - 1)
        self.water_if_empty(r, c + 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r + 1, c)
        self.water_if_empty(r + 1, c + 1)

    def water_circle(self, r, c):
        """Figura:
        .   .   .
        .   C   .
        .   .   .
        """
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r - 1, c)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r, c - 1)
        self.water_if_empty(r, c + 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r + 1, c)
        self.water_if_empty(r + 1, c + 1)

    def water_middle(self, r, c):
        """Preenche diagonais com água"""
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r + 1, c + 1)

    def water_top(self, r, c):
        """Figura:
        .   .   .
        .   T   .
        .       .
        """
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r - 1, c)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r, c - 1)
        self.water_if_empty(r, c + 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r + 1, c + 1)

    def water_left(self, r, c):
        """Figura:
        .   .   .
        .   L
        .   .   .
        """
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r - 1, c)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r, c - 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r + 1, c)
        self.water_if_empty(r + 1, c + 1)

    def water_right(self, r, c):
        """Figura:
        .   .   .
            R   .
        .   .   .
        """
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r - 1, c)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r, c + 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r + 1, c)
        self.water_if_empty(r + 1, c + 1)

    def water_infer_full_lines(self):
        for i in range(10):
            if self.ships_in_row(i) >= self.count_row[i]:
                for j in range(10):
                    self.water_if_empty(i, j)
            if self.ships_in_column(i) >= self.count_column[i]:
                for j in range(10):
                    self.water_if_empty(j, i)

    def water_all_empty(self):
        for r in range(10):
            for c in range(10):
                if self.get_value(r, c) == "":
                    self.grid[r, c] = "."

    def water_fill(self):
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if self.get_value(r, c).lower() == "w":
                    pass
                elif self.get_value(r, c).lower() == "c":
                    self.water_circle(r, c)
                elif self.get_value(r, c).lower() == "t":
                    self.water_top(r, c)
                elif self.get_value(r, c).lower() == "m":
                    self.water_middle(r, c)
                elif self.get_value(r, c).lower() == "b":
                    self.water_bottom(r, c)
                elif self.get_value(r, c).lower() == "l":
                    self.water_left(r, c)
                elif self.get_value(r, c).lower() == "r":
                    self.water_right(r, c)

    def check_diagonals(self, r, c) -> bool:
        return not (
            self.ship_in_cell(r - 1, c - 1)
            or self.ship_in_cell(r - 1, c + 1)
            or self.ship_in_cell(r + 1, c - 1)
            or self.ship_in_cell(r + 1, c + 1)
        )

    def ship_in_cell(self, r, c) -> bool:
        if not Board.within_bounds(r, c):
            return False
        temp = self.get_value(r, c)
        return temp not in {"", ".", "None", "W"}

    @staticmethod
    def ship_in_cell_value(v: str) -> bool:
        return v not in {"", ".", "None", "W"}

    def perpendicular_intersection(self, r, c) -> bool:
        if self.ship_in_cell(r, c):
            return (self.ship_in_cell(r - 1, c) or self.ship_in_cell(r + 1, c)) and (
                self.ship_in_cell(r, c - 1) or self.ship_in_cell(r, c + 1)
            )
        return False

    def ships_in_row(self, r) -> int:
        count = 0
        for i in range(10):
            if self.ship_in_cell(r, i):
                count += 1
        return count

    def ships_in_column(self, c) -> int:
        count = 0
        for i in range(10):
            if self.ship_in_cell(i, c):
                count += 1
        return count

    def ship_hint_in_cell(self, r, c) -> bool:
        cell = self.get_value(r, c).lower()
        return cell in {"t", "l", "r", "b", "m"}

    def cells_existing_in_new_ship(self, r, c, size, direction) -> int:
        count = 0
        for i in range(size):
            if self.ship_hint_in_cell(r + i * (1 - direction), c + i * direction):
                count += 1
        return count

    def check_action(self, r: int, c: int, size: int, direction: bool) -> bool:
        if self.ships[size - 1] >= 5 - size:
            Board.debug_action_checks(
                r, c, size, direction, "Demasiados navios deste tamanho"
            )
            return False

        if Board.within_bounds(
            r + (size - 1) * (1 - direction), c + (size - 1) * (direction)
        ):
            if size == 1:
                if (
                    self.ship_in_cell(r - 1, c)
                    or self.ship_in_cell(r + 1, c)
                    or self.ship_in_cell(r, c - 1)
                    or self.ship_in_cell(r, c + 1)
                    or not self.check_diagonals(r, c)
                    or self.ship_in_cell(r, c)
                    or self.is_water(r, c)
                    or self.ships_in_row(r) + 1 > self.count_row[r]
                    or self.ships_in_column(c) + 1 > self.count_column[c]
                ):
                    return False
                else:
                    return True

            if direction == HORIZONTAL:
                if self.get_value(r, c) == "l":
                    Board.debug_action_checks(
                        r, c, size, direction, "Navio já presente"
                    )
                    return False

                new_count = (
                    self.ships_in_row(r)
                    + size
                    - self.cells_existing_in_new_ship(r, c, size, direction)
                )
                if new_count > self.count_row[r]:
                    Board.debug_action_checks(
                        r, c, size, direction, f"Linha excederia ajudas ({new_count})"
                    )
                    return False

                if self.ship_in_cell(r, c - 1) or self.ship_in_cell(r, c + size):
                    Board.debug_action_checks(r, c, size, direction, "Navio adjacente")
                    return False

                for j in range(size):
                    new_count = (
                        self.ships_in_column(c + j)
                        + 1
                        - int(self.ship_in_cell(r, c + j))
                    )
                    if new_count > self.count_column[c + j]:
                        Board.debug_action_checks(
                            r,
                            c,
                            size,
                            direction,
                            f"Coluna {c + j} excederia ajudas ({new_count})",
                        )
                        return False

                existe = True
                for i in range(size):
                    cell = self.get_value(r, c + i)

                    if cell.lower() == "c":
                        return False

                    if self.is_water(r, c + i):
                        Board.debug_action_checks(r, c, size, direction, "Agua")
                        return False

                    if not self.check_diagonals(r, c + i):
                        Board.debug_action_checks(
                            r, c, size, direction, "Diagonais contêm navio"
                        )
                        return False

                    if cell.lower() == "r" and i < size - 1:
                        Board.debug_action_checks(
                            r,
                            c,
                            size,
                            direction,
                            "Navio chega ao fim antes do tamanho especificado pela função",
                        )
                        return False

                    if cell.lower() == "l" and i > 0:
                        Board.debug_action_checks(
                            r, c, size, direction, f"Overlap na posição {r} {c + i}"
                        )
                        return False

                    if cell.lower() in {"t", "b"}:
                        Board.debug_action_checks(
                            r, c, size, direction, f"Overlap na posição {r + i} {c}"
                        )
                        return False

                    if cell.lower() == "m" and (i < 1 or i == size - 1):
                        Board.debug_action_checks(
                            r, c, size, direction, "M no inicio ou fim"
                        )
                        return False

                    if not self.ship_in_cell(r, c + i):
                        existe = False
                if existe:
                    return False

            elif direction == VERTICAL:
                if self.get_value(r, c) == "t":
                    Board.debug_action_checks(
                        r, c, size, direction, "Navio já presente"
                    )
                    return False

                new_count = (
                    self.ships_in_column(c)
                    + size
                    - self.cells_existing_in_new_ship(r, c, size, direction)
                )
                if new_count > self.count_column[c]:
                    Board.debug_action_checks(
                        r, c, size, direction, f"Coluna excederia ajudas ({new_count})"
                    )
                    return False

                if self.ship_in_cell(r - 1, c) or self.ship_in_cell(r + size, c):
                    Board.debug_action_checks(r, c, size, direction, "Navio adjacente")
                    return False

                for j in range(size):
                    new_count = (
                        self.ships_in_row(r + j) + 1 - int(self.ship_in_cell(r + j, c))
                    )
                    if new_count > self.count_row[r + j]:
                        Board.debug_action_checks(
                            r,
                            c,
                            size,
                            direction,
                            f"Linha {r+j} excederia ajudas ({new_count})",
                        )
                        return False

                existe = True
                for i in range(size):
                    cell = self.get_value(r + i, c)

                    if cell.lower() == "c":
                        return False

                    if not self.check_diagonals(r + i, c):
                        Board.debug_action_checks(
                            r, c, size, direction, "Diagonais contêm navio"
                        )
                        return False

                    if self.is_water(r + i, c):
                        Board.debug_action_checks(r, c, size, direction, "Agua")
                        return False

                    if cell.lower() == "b" and i < size - 1:
                        Board.debug_action_checks(
                            r,
                            c,
                            size,
                            direction,
                            "Navio chega ao fim antes do tamanho especificado pela função",
                        )
                        return False

                    if cell.lower() == "t" and i > 0:
                        Board.debug_action_checks(
                            r, c, size, direction, f"Overlap na posição {r + i} {c}"
                        )
                        return False

                    if cell.lower() == "m" and (i < 1 or i == size - 1):
                        Board.debug_action_checks(
                            r, c, size, direction, "M no inicio ou fim"
                        )
                        return False

                    if cell.lower() in {"l", "r"}:
                        Board.debug_action_checks(
                            r, c, size, direction, f"Overlap na posição {r + i} {c}"
                        )
                        return False

                    if not self.ship_in_cell(r + i, c):
                        existe = False
                if existe:
                    Board.debug_action_checks(
                        r, c, size, direction, "Navio total pelas hints"
                    )
                    return False
            return True

        Board.debug_action_checks(r, c, size, direction, "Out of bounds")
        return False

    @staticmethod
    def debug_action_checks(r: int, c: int, size: int, direction: bool, text: str):
        if DEBUG_LEVEL == D_VERBOSE:
            print("* ", r, c, size, direction, text, sep="\t")

    def print_debug(self) -> str:
        buf: str = "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n"
        for r in range(10):
            buf += str(r) + "\t"
            for c in range(10):
                buf += self.get_value(r, c)
                if c < 9:
                    buf += "\t"
            buf += "\t" + str(self.count_row[r]) + "\n"
        for x in self.count_column:
            buf += "\t" + str(x)
        buf += "\n\n"
        return buf

    def print(self):
        for r in range(10):
            for c in range(10):
                print(self.get_value(r, c), end="", sep="")
            print("")

    # TODO: outros metodos da classe


class BimaruState:
    state_id = 0

    def __init__(self, board: Board, ghost: bool = False):
        self.board = board
        if not ghost:
            self.id = BimaruState.state_id
            BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return np.array_equal(self.board.grid, other.board.grid)

    def __hash__(self):
        return hash(str(self.board.grid))


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if not state.board.valid_board():
            return []

        a = []
        s = -1

        for i in reversed(range(1, 5)):
            if state.board.ships[i - 1] < 5 - i:
                s = i
                break
        if s == -1:
            return []

        for r in range(10):
            for c in range(10):
                for d in range(2):
                    if s != 1:
                        if state.board.check_action(r, c, s, d):
                            a.append((r, c, s, d))
                    else:
                        if state.board.check_action(r, c, s, 0):
                            a.append((r, c, s, 0))
                        break

        n_actions = len(a)
        if state.board.ships[s - 1] + n_actions < 5 - s:
            # Ações insuficientes para completar os navios necessários
            return []
        elif state.board.ships[s - 1] + n_actions == 5 - s:
            # As ações têm de ser mutuamente compatíveis, pelo que só
            # vale a pena seguir um ramo (se forem, aparecerão no filho)
            return [a[0]]

        return a

    def result(self, state: BimaruState, action: Action) -> BimaruState:
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = Board(
            np.array(state.board.grid, str),
            np.array(state.board.ships, int),
            np.array(state.board.count_row, int),
            np.array(state.board.count_column, int),
        )
        new_state = BimaruState(new_board)
        if DEBUG_LEVEL >= D_MINIMUM:
            buf: str = ""
            buf += "―" * 90 + "\n"
            buf += f"from state {state.id} with {action}\n"

        row = action[A_ROW]
        col = action[A_COL]
        ship_size = action[A_SIZE]
        direction = action[A_DIR]

        if ship_size == 1:
            new_state.board.grid[row, col] = "c"
        else:
            if direction == HORIZONTAL:
                new_state.board.set_if_empty_and_within_bounds(row, col, "l")
                new_state.board.set_if_empty_and_within_bounds(
                    row, col + ship_size - 1, "r"
                )
            else:
                new_state.board.set_if_empty_and_within_bounds(row, col, "t")
                new_state.board.set_if_empty_and_within_bounds(
                    row + ship_size - 1, col, "b"
                )
            for i in range(1, ship_size - 1):
                new_state.board.set_if_empty_and_within_bounds(
                    row + i * (1 - direction), col + i * direction, "m"
                )

        new_state.board.water_fill()
        new_state.board.ships[ship_size - 1] += 1
        new_state.board.inferences()
        new_state.board.ship_count()

        if DEBUG_LEVEL >= D_MINIMUM:
            buf += f"to state {new_state.id} :\n"
            buf += new_state.board.print_debug()
            buf += "ships : " + str(new_state.board.ships) + "\n"
            actions = self.actions(new_state)
            buf += "actions: " + str(actions) + "\n"
            # buf += "valid: " + str(new_state.board.valid_board())
            print(buf)
        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # print(state.id, state.board.valid_board(), state.board.ships)
        return (
            np.array_equal(state.board.ships, [4, 3, 2, 1])
            and state.board.valid_board()
            and state.board.pontas_soltas()
        )

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        ship_count = node.state.board.ships
        # return 20 - ship_count[0] * 1 - ship_count[1] * 2 - ship_count[2] * 3 - ship_count[3] * 3
        return 10 - ship_count[0] - ship_count[1] - ship_count[2] - ship_count[3]

    def debug_loop(self):
        action = ()
        new_state = self.initial
        while 1:
            action = stdin.readline().rstrip("\n").split(" ")[0:]
            for i in range(4):
                action[i] = int(action[i])
            print(action)
            new_state = self.result(new_state, action)
            print(new_state.board.valid_board())


def main():
    initial_board = Board.parse_instance()
    bimaru = Bimaru(initial_board)
    if DEBUG_LEVEL >= D_MINIMUM:
        print(bimaru.initial.board.print_debug())
        print("")
        print(bimaru.initial.board.ships)
        print("")
        print(bimaru.actions(bimaru.initial))

    # start = time.perf_counter()
    # goal_node = depth_first_tree_search(bimaru)
    goal_node = depth_first_graph_search(bimaru)
    # goal_node = astar_search(bimaru,bimaru.h)
    # end = time.perf_counter()

    # bimaru.debug_loop()

    if goal_node is None:
        print("No final state.")
        return
    goal_node.state.board.water_all_empty()
    if DEBUG_LEVEL >= D_MINIMUM:
        print("\t\t\t\t\tFINAL STATE", sep="")
        print(goal_node.state.board.print_debug())
    else:
        goal_node.state.board.print()


if __name__ == "__main__":
    main()
