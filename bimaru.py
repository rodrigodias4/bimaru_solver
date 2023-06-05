# Grupo 135:
# 99552 Rodrigo Dias

import sys
from sys import stdin
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

HORIZONTAL = 1
VERTICAL = 0
D_NONE = 0
D_MINIMUM = 1
D_VERBOSE = 2
A_ROW = 0
A_COL = 1
A_SIZE = 2
A_DIR = 3
DEBUG_LEVEL = D_NONE


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    # Variáveis estáticas, já que todos os tabuleiros
    # têm as mesmas dicas
    count_row: np.ndarray
    count_column: np.ndarray

    def __init__(
        self,
        grid: np.ndarray,
        ships: np.ndarray,
    ):
        self.grid = grid
        self.ships = ships

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor da célula"""
        if not Board.within_bounds(row, col):
            return ""
        return self.grid[row, col]

    @staticmethod
    def within_bounds(r: np.byte, c: np.byte) -> bool:
        """Retorna se (r,c) é uma posição válida"""
        return 0 <= r < 10 and 0 <= c < 10

    def is_water(self, r: np.byte, c: np.byte) -> bool:
        """Retorna true se a célula contém água"""
        v = self.get_value(r, c)
        return v in {".", "W"}

    def is_water_value(self, v) -> bool:
        """Retorna true se a string v corresponde a água"""
        return v in {".", "W"}

    def is_water_or_oob(self, r: np.byte, c: np.byte):
        """Verifica se a célula é água ou está fora dos limites"""
        return not Board.within_bounds(r, c) or self.is_water(r, c)

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        above = "None"
        below = "None"

        if Board.within_bounds(row - 1, col):
            v = self.get_value(row - 1, col)
            if v != "":
                above = v
        if Board.within_bounds(row + 1, col):
            v = self.get_value(row + 1, col)
            if v != "":
                below = v

        return (above, below)

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left = "None"
        right = "None"

        if Board.within_bounds(row, col - 1):
            v = self.get_value(row, col - 1)
            if v != "":
                left = v
        if Board.within_bounds(row, col + 1):
            v = self.get_value(row, col + 1)
            if v != "":
                right = v

        return (left, right)

    def ship_in_cell(self, r: np.byte, c: np.byte) -> bool:
        """Retorna se existe um navio na célula"""
        temp = self.get_value(r, c)
        return temp not in {"", ".", "None", "W"}

    @staticmethod
    def is_ship(v: str) -> bool:
        """Retorna se v corresponde a um navio"""
        return v not in {"", ".", "None", "W"}

    def water_if_empty(self, r: np.byte, c: np.byte):
        """Coloca água na célula se esta estiver vazia"""
        self.set_if_empty(r, c, ".")

    def set_if_empty(self, r: np.byte, c: np.byte, value: str):
        """Coloca o valor indicado na célula se esta estiver vazia"""
        if Board.within_bounds(r, c):
            if self.grid[r, c] == "":
                self.grid[r, c] = value

    def inferences_middle(self, r: np.byte, c: np.byte):
        """Inferências para uma célula que contém m"""
        # M numa linha/coluna inicial ou final corresponde
        # a um navio paralelo a essa borda do tabuleiro
        if r == 0:
            self.water_if_empty(r + 1, c)
        elif r == 9:
            self.water_if_empty(r - 1, c)
        if c == 0:
            self.water_if_empty(r, c + 1)
        elif c == 9:
            self.water_if_empty(r, c - 1)

        (above, below) = self.adjacent_vertical_values(r, c)
        (above, below) = (above.lower(), below.lower())
        (left, right) = self.adjacent_horizontal_values(r, c)
        (left, right) = (left.lower(), right.lower())

        if self.ship_cells_in_row(r) + 2 > self.count_row[r] and not (
            left == "l" or right == "r" or "m" in {left, right}
        ):
            # Se o número de células com navios na linha exceder
            # as ajudas + 2 (células necessárias para fazer um navio
            # com pelo menos um M) significa que o navio a que esta célula m
            # pertence tem de estar na direção perpendicular,
            # logo, insere-se água nas posições adjacentes da linha
            self.water_if_empty(r, c - 1)
            self.water_if_empty(r, c + 1)
        elif self.ship_cells_in_column(c) + 2 > self.count_column[c] and not (
            above == "t" or below == "b" or "m" in {above, below}
        ):
            # Idêntico ao caso anterior, aplicado a colunas
            self.water_if_empty(r - 1, c)
            self.water_if_empty(r + 1, c)

        if (self.ship_cells_in_row(r) + 2 - (right == "r") - (left == "l")) == self.count_row[
            r
        ] and (self.is_water_or_oob(r - 1, c) or self.is_water_or_oob(r + 1, c)):
            #   .                        .
            # []M[] (2 left on row) ->  lMr
            #   .                        .
            self.set_if_empty(r, c - 1, "l")
            self.set_if_empty(r, c + 1, "r")
        elif self.ship_cells_in_column(c) + 2 == self.count_row[c] and (
            self.is_water_or_oob(r, c - 1) or self.is_water_or_oob(r, c + 1)
        ):
            # Perpendicular ao anterior
            self.set_if_empty(r - 1, c, "t")
            self.set_if_empty(r + 1, c, "b")

        if self.ship_cells_in_row(r) + 1 == self.count_row[r]:
            if left == "l" and self.get_value(r, c + 2).lower() != "r":
                # LM[] -> LMr, caso falte apenas uma célula preenchida
                # para o número de ajudas e não tenhamos LM[]R
                self.set_if_empty(r, c + 1, "r")
            elif right == "r" and self.get_value(r, c - 2).lower() != "l":
                # Simétrico do anterior
                self.set_if_empty(r, c - 1, "l")
        elif self.ship_cells_in_column(c) + 1 == self.count_column[c]:
            # Perpendicular ao anterior
            if above == "t" and self.get_value(r + 2, c).lower() != "b":
                self.set_if_empty(r + 1, c, "b")
            elif below == "b" and self.get_value(r - 2, c).lower() != "t":
                self.set_if_empty(r - 1, c, "t")

        if left == "m":
            # MM[] -> MMr
            self.set_if_empty(r, c + 1, "r")
        elif right == "m":
            self.set_if_empty(r, c - 1, "l")
        if above == "m":
            self.set_if_empty(r + 1, c, "b")
        elif below == "m":
            self.set_if_empty(r - 1, c, "t")

        # Água adjacente a M significa que o navio tem de ser
        # perpendicular à posição da água
        if self.is_water(r, c - 1) or self.is_water(r, c + 1):
            if self.is_water_or_oob(r - 2, c):
                # .    .
                #   -> t
                # M    M
                self.set_if_empty(r - 1, c, "t")
            if self.is_water_or_oob(r + 2, c):
                # Simétrico do anterior
                self.set_if_empty(r + 1, c, "b")
            self.water_if_empty(r, c - 1)
            self.water_if_empty(r, c + 1)
        elif self.is_water(r - 1, c) or self.is_water(r + 1, c):
            # Perpendicular ao anterior (caso horizontal)
            if self.is_water_or_oob(r, c - 2):
                self.set_if_empty(r, c - 1, "l")
            if self.is_water_or_oob(r, c + 2):
                self.set_if_empty(r, c + 1, "r")
            self.water_if_empty(r - 1, c)
            self.water_if_empty(r + 1, c)

    def inferences_aux(self, v: str) -> (np.byte, bool, np.byte):
        if v == "t":
            op = "b"
            d = VERTICAL
            s = 1
        elif v == "b":
            op = "t"
            d = VERTICAL
            s = -1
        elif v == "l":
            op = "r"
            d = HORIZONTAL
            s = 1
        elif v == "r":
            op = "l"
            d = HORIZONTAL
            s = -1
        else:
            return

        return (op, d, s)

    def inferences_template(self, r: np.byte, c: np.byte, v):
        """Abstrai as inferências de T, B, L e R a uma função,
        de modo a reduzir a extensão do código
        op: valor da ponta oposta do navio
        d: direção
        s: sentido"""
        v = v.lower()
        (op, d, s) = self.inferences_aux(v)

        if self.is_water_or_oob(r + 2 * (1 - d) * s, c + 2 * d * s):
            # CONTRATORPEDEIRO (2 CELULAS) | L[]. -> Lr.
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, op)
        elif (
            self.ship_cells_in_column(c) * (1 - d) + self.ship_cells_in_row(r) * d + 1
            == self.count_column[c] * (1 - d) + self.count_row[r] * d
        ) and not self.ship_in_cell(r + 2 * (1 - d) * s, c + 2 * d * s):
            # CONTRATORPEDEIRO (2 CELULAS) | L[] (1 left on row) -> Lr
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, op)
        elif self.get_value(r + 3 * (1 - d) * s, c + 3 * d * s).lower() == "m":
            # CONTRATORPEDEIRO (2 CELULAS) e OUTRO NAVIO
            # L[]?M -> Lr.M
            self.water_if_empty(r + 2 * (1 - d) * s, c + 2 * d * s)
            self.water_if_empty(r + 4 * (1 - d) * s, c + 4 * d * s)
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, op)
        elif self.get_value(r + 2 * (1 - d) * s, c + 2 * d * s).lower() == op:
            # CRUZADOR (3 CELULAS) | L[]R -> LmR
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, "m")
        elif self.get_value(
            r + 1 * (1 - d) * s, c + 1 * d * s
        ).lower() == "m" and self.is_water_or_oob(r + 3 * (1 - d) * s, c + 3 * d * s):
            # CRUZADOR (3 CELULAS) | LM[]. -> LMr
            self.set_if_empty(r + 2 * (1 - d) * s, c + 2 * d * s, op)
        elif self.get_value(
            r + 2 * (1 - d) * s, c + 2 * d * s
        ).lower() == "m" and self.is_water_or_oob(r + 4 * (1 - d) * s, c + 4 * d * s):
            # COURAÇADO (4 CELULAS) | L[]M[]. -> LmMr
            self.set_if_empty(r + 3 * (1 - d) * s, c + 3 * d * s, op)
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, "m")
        elif self.get_value(r + 3 * (1 - d) * s, c + 3 * d * s) == op:
            # COURAÇADO (4 CELULAS) | L[][]R -> LmmR
            self.set_if_empty(r + 1 * (1 - d) * s, c + 1 * d * s, "m")
            self.set_if_empty(r + 2 * (1 - d) * s, c + 2 * d * s, "m")

    def inferences_top(self, r: np.byte, c: np.byte):
        self.inferences_template(r, c, "t")

    def inferences_bottom(self, r: np.byte, c: np.byte):
        self.inferences_template(r, c, "b")

    def inferences_left(self, r: np.byte, c: np.byte):
        self.inferences_template(r, c, "l")

    def inferences_right(self, r: np.byte, c: np.byte):
        self.inferences_template(r, c, "r")

    def inferences(self):
        """Infere navios obrigatórios e linhas cheias
        com base no tabuleiro atual e hints"""
        self.inferences_water_full_lines()

        for r in range(10):
            for c in range(10):
                v = self.get_value(r, c).lower()
                if v == "m":
                    self.inferences_middle(r, c)
                elif v == "t":
                    self.inferences_top(r, c)
                elif v == "b":
                    self.inferences_bottom(r, c)
                elif v == "l":
                    self.inferences_left(r, c)
                elif v == "r":
                    self.inferences_right(r, c)

        self.water_adjacences()  # Preenche adjacências de navios com água
        self.inferences_water_full_lines()

    def inferences_water_full_lines(self):
        """Prenche com água os espaços vazios de linhas/colunas
        com número de células com navio igual à dica correspondente"""
        for i in range(10):
            if self.ship_cells_in_row(i) >= self.count_row[i]:
                for j in range(10):
                    self.water_if_empty(i, j)
            if self.ship_cells_in_column(i) >= self.count_column[i]:
                for j in range(10):
                    self.water_if_empty(j, i)

    def water_adjacences(self):
        """Infere e preenche água em posições adjacentes a navios"""
        for r in range(10):
            for c in range(10):
                v = self.get_value(r, c).lower()
                if v == "w":
                    pass
                elif v == "c":
                    self.water_circle(r, c)
                elif v == "t":
                    self.water_top(r, c)
                elif v == "m":
                    self.water_middle(r, c)
                elif v == "b":
                    self.water_bottom(r, c)
                elif v == "l":
                    self.water_left(r, c)
                elif v == "r":
                    self.water_right(r, c)

    @staticmethod
    def parse_instance():
        """Retorna um tabuleiro com base no input formatado"""

        # Criação de variáveis do tabuleiro
        initial_grid = np.empty([10, 10], "U1")
        Board.count_row = np.array(
            stdin.readline().rstrip("\n").split("\t")[1:], np.byte
        )
        Board.count_column = np.array(
            stdin.readline().rstrip("\n").split("\t")[1:], np.byte
        )

        n_hints = int(stdin.readline().rstrip("\n"))  # Número de dicas

        # Lê cada uma das dicas e insere no tabuleiro
        for i in range(n_hints):
            line = stdin.readline().rstrip("\n").split("\t")[1:]
            initial_grid[int(line[0]), int(line[1])] = line[2]

        initial_board = Board(initial_grid, np.zeros(4, np.ubyte))

        initial_board.water_adjacences()  # Infere água a partir das dicas

        # Infere até não haver mais alterações por inferência
        while True:
            previous_board = Board(
                np.array(initial_board.grid, "U1"),
                np.array(initial_board.ships, np.byte),
            )
            initial_board.inferences()
            if initial_board == previous_board:
                break

        initial_board.ship_count()  # Conta os navios presentes

        return initial_board

    def ship_count(self):
        """Conta os navios presentes manualmente"""
        self.ships = np.zeros(4, np.ubyte)

        for r in range(10):
            for c in range(10):
                v = self.get_value(r, c).lower()
                if v == "c":
                    self.ships[0] += 1
                elif v == "t":
                    for i in range(1, 4):
                        vi = self.get_value(r + i, c).lower()
                        if vi == "b":
                            self.ships[i] += 1
                        if vi != "m":
                            break
                elif v == "l":
                    for i in range(1, 4):
                        vi = self.get_value(r, c + i).lower()
                        if vi == "r":
                            self.ships[i] += 1
                            c += i
                        if vi != "m":
                            break

    def pontas_soltas(self) -> bool:
        """Verifica se existe células preenchidas que não
        correspondem a um navio (i.e. dicas não usadas)"""
        for r in range(10):
            for c in range(10):
                v = self.get_value(r, c).lower()
                if v == "t":
                    temp = self.get_value(r + 1, c).lower()
                    if temp not in {"m", "b"}:  # T isolado
                        return False
                elif v == "b":
                    temp = self.get_value(r - 1, c).lower()
                    if temp not in {"m", "t"}:  # B isolado
                        return False
                elif v == "l":
                    temp = self.get_value(r, c + 1).lower()
                    if temp not in {"m", "r"}:  # L isolado
                        return False
                elif v == "r":
                    temp = self.get_value(r, c - 1).lower()
                    if temp not in {"m", "l"}:  # R isolado
                        return False
                elif v == "m":
                    adjacent_to_m = (
                        int(self.is_water_or_oob(r - 1, c))
                        + int(self.is_water_or_oob(r + 1, c))
                        + int(self.is_water_or_oob(r, c - 1))
                        + int(self.is_water_or_oob(r, c + 1))
                    )
                    if adjacent_to_m != 2:
                        # Uma célula com M tem de ter o resto
                        # do navio numa direção e água (ou fora
                        # do tabuleiro) na outra
                        if DEBUG_LEVEL == D_VERBOSE:
                            print(f"m {r} {c}")
                        return False

        # Verifica se as ajudas de linhas e colunas correspondem
        # às contagens finais correspondentes
        for i in range(10):
            if self.ship_cells_in_row(i) != self.count_row[i]:
                return False
            if self.ship_cells_in_column(i) != self.count_column[i]:
                return False
        return True

    def water_bottom(self, r: np.byte, c: np.byte):
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

    def water_circle(self, r: np.byte, c: np.byte):
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

    def water_middle(self, r: np.byte, c: np.byte):
        """Preenche diagonais com água"""
        self.water_if_empty(r - 1, c - 1)
        self.water_if_empty(r + 1, c - 1)
        self.water_if_empty(r - 1, c + 1)
        self.water_if_empty(r + 1, c + 1)

    def water_top(self, r: np.byte, c: np.byte):
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

    def water_left(self, r: np.byte, c: np.byte):
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

    def water_right(self, r: np.byte, c: np.byte):
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

    def water_all_empty(self):
        """Insere água em todas as posições vazias do tabuleiro"""
        for r in range(10):
            for c in range(10):
                if self.get_value(r, c) == "":
                    self.grid[r, c] = "."

    def check_diagonals(self, r: np.byte, c: np.byte) -> bool:
        return not (
            self.ship_in_cell(r - 1, c - 1)
            or self.ship_in_cell(r - 1, c + 1)
            or self.ship_in_cell(r + 1, c - 1)
            or self.ship_in_cell(r + 1, c + 1)
        )

    def ship_cells_in_row(self, r) -> int:
        """Conta o número de células com navio na linha"""
        count = 0
        for i in range(10):
            if self.ship_in_cell(r, i):
                count += 1
        return count

    def ship_cells_in_column(self, c) -> int:
        """Conta o número de células com navio na coluna"""
        count = 0
        for i in range(10):
            if self.ship_in_cell(i, c):
                count += 1
        return count

    def cells_existing_in_new_ship(
        self, r: np.byte, c: np.byte, size, direction
    ) -> int:
        """Verifica o número de células já ocupadas no navio a inserir"""
        count = 0
        for i in range(size):
            if self.ship_in_cell(r + i * (1 - direction), c + i * direction):
                count += 1
        return count

    def check_action_size_one(self, r: np.byte, c: np.byte) -> bool:
        """Auxiliar de check_action para navios de tamanho 1"""
        if (
            self.ship_in_cell(r - 1, c)
            or self.ship_in_cell(r + 1, c)
            or self.ship_in_cell(r, c - 1)
            or self.ship_in_cell(r, c + 1)
            or not self.check_diagonals(r, c)
            or self.ship_in_cell(r, c)
            or self.is_water(r, c)
            or self.ship_cells_in_row(r) + 1 > self.count_row[r]
            or self.ship_cells_in_column(c) + 1 > self.count_column[c]
        ):
            return False
        return True

    def check_action_horizontal(self, r: np.byte, c: np.byte, size: int) -> bool:
        """Auxiliar de check_action para ações horizontais"""

        new_count = (
            self.ship_cells_in_row(r)
            + size
            - self.cells_existing_in_new_ship(r, c, size, HORIZONTAL)
        )
        if new_count > self.count_row[r]:
            Board.debug_action_checks(
                r, c, size, HORIZONTAL, f"Linha excederia ajudas ({new_count})"
            )
            return False

        if self.ship_in_cell(r, c - 1) or self.ship_in_cell(r, c + size):
            Board.debug_action_checks(r, c, size, HORIZONTAL, "Navio na ponta")
            return False

        existe = True
        for i in range(size):
            cell = self.get_value(r, c + i).lower()

            if cell == "c":
                Board.debug_action_checks(r, c, size, HORIZONTAL, "C")
                return False
            if self.is_water_value(cell):
                Board.debug_action_checks(r, c, size, HORIZONTAL, "Agua")
                return False

            if not self.check_diagonals(r, c + i):
                Board.debug_action_checks(
                    r, c, size, HORIZONTAL, "Diagonais contêm navio"
                )
                return False

            if cell == "r" and i < size - 1:
                Board.debug_action_checks(
                    r,
                    c,
                    size,
                    HORIZONTAL,
                    "Navio chega ao fim antes do tamanho especificado pela função",
                )
                return False

            if cell == "l" and i > 0:
                Board.debug_action_checks(
                    r, c, size, HORIZONTAL, f"Overlap na posição {r} {c + i}"
                )
                return False

            if cell in {"t", "b"}:
                Board.debug_action_checks(
                    r, c, size, HORIZONTAL, f"Overlap na posição {r + i} {c}"
                )
                return False

            if cell == "m" and (i < 1 or i == size - 1):
                Board.debug_action_checks(r, c, size, HORIZONTAL, "M no inicio ou fim")
                return False

            if not self.is_ship(cell):
                existe = False
        if existe:
            Board.debug_action_checks(r, c, size, HORIZONTAL, "Navio total pelas hints")
            return False
        return True

    def check_action_vertical(self, r: np.byte, c: np.byte, size: int) -> bool:
        """Auxiliar de check_action para ações verticais"""
        new_count = (
            self.ship_cells_in_column(c)
            + size
            - self.cells_existing_in_new_ship(r, c, size, VERTICAL)
        )
        if new_count > self.count_column[c]:
            Board.debug_action_checks(
                r, c, size, VERTICAL, f"Coluna excederia ajudas ({new_count})"
            )
            return False

        if self.ship_in_cell(r - 1, c) or self.ship_in_cell(r + size, c):
            Board.debug_action_checks(r, c, size, VERTICAL, "Navio na ponta")
            return False

        existe = True
        for i in range(size):
            cell = self.get_value(r + i, c).lower()

            if cell == "c":
                Board.debug_action_checks(r, c, size, VERTICAL, "C")
                return False

            if not self.check_diagonals(r + i, c):
                Board.debug_action_checks(
                    r, c, size, VERTICAL, "Diagonais contêm navio"
                )
                return False

            if self.is_water_value(cell):
                Board.debug_action_checks(r, c, size, VERTICAL, "Agua")
                return False

            if cell == "b" and i < size - 1:
                Board.debug_action_checks(
                    r,
                    c,
                    size,
                    VERTICAL,
                    "Navio chega ao fim antes do tamanho especificado pela função",
                )
                return False

            if cell == "t" and i > 0:
                Board.debug_action_checks(
                    r, c, size, VERTICAL, f"Overlap na posição {r + i} {c}"
                )
                return False

            if cell == "m" and (i < 1 or i == size - 1):
                Board.debug_action_checks(r, c, size, VERTICAL, "M no inicio ou fim")
                return False

            if cell in {"l", "r"}:
                Board.debug_action_checks(
                    r, c, size, VERTICAL, f"Overlap na posição {r + i} {c}"
                )
                return False

            if not self.is_ship(cell):
                existe = False
        if existe:
            Board.debug_action_checks(r, c, size, VERTICAL, "Navio total pelas hints")
            return False
        return True

    def check_action(self, r: np.byte, c: np.byte, size: int, direction: bool) -> bool:
        """Verifica se a ação resulta num tabuleiro válido"""
        if self.ships[size - 1] >= 5 - size:
            Board.debug_action_checks(
                r, c, size, direction, "Demasiados navios deste tamanho"
            )
            return False

        if Board.within_bounds(
            r + (size - 1) * (1 - direction), c + (size - 1) * (direction)
        ):
            if size == 1:
                return self.check_action_size_one(r, c)

            if direction == HORIZONTAL:
                if not self.check_action_horizontal(r, c, size):
                    return False
            elif direction == VERTICAL:
                if not self.check_action_vertical(r, c, size):
                    return False

            Board.debug_action_checks(r, c, size, direction, "AÇÃO VÁLIDA")
            return True

        Board.debug_action_checks(r, c, size, direction, "Out of bounds")
        return False

    @staticmethod
    def debug_action_checks(
        r: np.byte, c: np.byte, size: int, direction: bool, text: str
    ):
        """Debugging do método actions()"""
        if DEBUG_LEVEL == D_VERBOSE:
            print("Action", r, c, size, direction, text, sep="\t")

    def print_debug(self) -> str:
        """Apenas para debugging:
        Imprime o tabuleiro num formato mais estético e com mais informações"""
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
        """Imprime o tabuleiro segundo o formato de output pedido"""
        for r in range(10):
            for c in range(10):
                print(self.grid[r, c], end="", sep="")
            print("")

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    def __hash__(self):
        return hash(str(self.grid))


class BimaruState:
    state_id = 0

    def __init__(self, board: Board):
        self.board = board
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
        super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        a = np.empty(100, (np.byte, 4))
        s = -1
        n_actions = 0

        # Escolhe o tamanho maior que ainda não tem os navios todos
        for i in reversed(range(1, 5)):
            if state.board.ships[i - 1] < 5 - i:
                s = i
                break
        if s == -1:
            return []

        # Verifica cada ação e adiciona à lista se esta
        # resulta num tabuleiro válido
        for r in range(10):
            for c in range(10):
                for d in range(2):
                    if s != 1:
                        if state.board.check_action(r, c, s, d):
                            n_actions += 1
                            even = 1 - (n_actions % 2)
                            i = (
                                50
                                + int(even * (n_actions / 2))
                                - int(np.floor((not even) * n_actions / 2))
                            )
                            a[i] = np.array([r, c, s, d], np.byte)
                    else:
                        # Evita ações de tamanho 1 duplicadas
                        # (direção redundante)
                        if state.board.check_action(r, c, s, 0):
                            n_actions += 1
                            even = 1 - (n_actions % 2)
                            i = (
                                50
                                + int(even * (n_actions / 2))
                                - int(np.floor((not even) * n_actions / 2))
                            )
                            a[i] = np.array([r, c, s, 0], np.byte)
                        break

        if state.board.ships[s - 1] + n_actions < 5 - s:
            # Ações insuficientes para completar os navios necessários
            return []
        elif state.board.ships[s - 1] + n_actions == 5 - s:
            # As ações têm de ser mutuamente compatíveis, pelo que só
            # vale a pena seguir um ramo (se forem, aparecerão no filho)
            return [a[50]]

        return a[
            50 - int(np.ceil(n_actions / 2 - 1)) : 50 + int(np.floor(n_actions / 2)) + 1
        ].tolist()

    def result(self, state: BimaruState, action: np.ndarray) -> BimaruState:
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = Board(
            np.array(state.board.grid, "U1"),
            np.array(state.board.ships, np.byte),
        )
        new_state = BimaruState(new_board)
        row = action[A_ROW]
        col = action[A_COL]
        ship_size = action[A_SIZE]
        direction = action[A_DIR]

        # Insere o navio
        if ship_size == 1:
            new_state.board.grid[row, col] = "c"
        else:
            if direction == HORIZONTAL:
                new_state.board.set_if_empty(row, col, "l")
                new_state.board.set_if_empty(row, col + ship_size - 1, "r")
            else:
                new_state.board.set_if_empty(row, col, "t")
                new_state.board.set_if_empty(row + ship_size - 1, col, "b")
            for i in range(1, ship_size - 1):
                new_state.board.set_if_empty(
                    row + i * (1 - direction), col + i * direction, "m"
                )

        # Infere até não haver mais alterações por inferência
        while True:
            previous_board = Board(
                np.array(new_state.board.grid, "U1"),
                np.array(new_state.board.ships, np.byte),
            )
            new_state.board.inferences()
            if new_state.board == previous_board:
                break
        new_state.board.ship_count()  # Reconta os navios

        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return (
            np.array_equal(state.board.ships, [4, 3, 2, 1])
            and state.board.pontas_soltas()
        )

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""

    def debug_loop(self):
        """Função estritamente para debugging,
        ações escolhidas manualmente"""
        action = ()
        new_state = self.initial
        print(new_state.board.print_debug())
        print(self.actions(new_state))
        while 1:
            action = stdin.readline().rstrip("\n").split(" ")[0:]
            for i in range(4):
                action[i] = int(action[i])
            print(action)
            new_state = self.result(new_state, action)
            print(new_state.board.print_debug())
            print(self.actions(new_state))


def main():
    # Interpreta a instância inicial a partir do stdin
    initial_board = Board.parse_instance()

    # Cria o problema a partir do tabuleiro inicial
    bimaru = Bimaru(initial_board)

    # Realiza a procura
    goal_node = depth_first_graph_search(bimaru)

    if goal_node is None:
        print("No final state.")
        return
    # Preenche os espaços vazios do tabuleiro final com água
    goal_node.state.board.water_all_empty()

    # Imprime o tabuleiro final
    goal_node.state.board.print()


if __name__ == "__main__":
    main()
