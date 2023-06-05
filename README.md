# Inteligência Artificial 2022/

# Projeto: Bimaru

### 2 de maio de 2023

## 1 Introdução

O projeto da unidade curricular de Inteligência Artificial (IA) tem como objetivo desenvolver
um programa em Python que resolva o problema Bimaru utilizando técnicas de IA.
O problema Bimaru, também denominado Puzzle Batalha Naval, Yubotu ou Batalha Naval
Solitário, é um puzzle inspirado no conhecido jogo de Batalha Naval entre dois jogadores.
O jogo foi criado na Argentina por Jaime Poniachik e apareceu pela primeira vez em 1982 na
revista argentinaHumor & Juegos. O jogo ficou conhecido internacionalmente ao ser integrado
pela primeira vez noWorld Puzzle Championshipem 1992.

## 2 Descrição do problema

De acordo com a descrição que consta na CSPlib^1 , o jogo Bimaru decorre numa grelha qua-
drada, representando uma área do oceano. Os jogos publicados geralmente usam uma grelha
de 10×10, pelo que assumiremos essa dimensão no contexto do projeto.
A área de oceano contém uma frota escondida que o jogador deve encontrar. Esta frota
consiste num couraçado (quatro quadrados de comprimento), dois cruzadores (cada um com
três quadrados de comprimento), três contratorpedeiros (cada um com dois quadrados de com-
primento) e quatro submarinos (um quadrado cada).
Os navios podem ser orientados horizontal ou verticalmente, e dois navios não ocupam
quadrados da grelha adjacentes, nem mesmo na diagonal. O jogador também recebe as con-
tagens de linha e coluna, ou seja, o número de quadrados ocupados em cada linha e coluna,
e várias dicas. Cada dica especifica o estado de um quadrado individual na grelha: água (o
quadrado está vazio); círculo (o quadrado é ocupado por um submarino); meio (este é um
quadrado no meio de um couraçado ou cruzador); superior, inferior, esquerda ou direita (este
quadrado é a extremidade de um navio que ocupa pelo menos dois quadrados).

## 3 Objetivo

O objetivo deste projeto é o desenvolvimento de um programa em Python 3.8 que, dada uma
instância de Bimaru, retorna uma solução, i.e., uma grelha totalmente preenchida.
O programa deve ser desenvolvido num ficheirobimaru.py, que lê uma instância de Bi-
maru a partir do standard input no formato descrito na secção 4.1. O programa deve resolver
o problema utilizando uma técnica à escolha e imprimir a solução para ostandard outputno
formato descrito na secção 4.2.

Utilização:

```
python3 bimaru.py < <instance_file>
```
## 4 Formato de input e output

O formato que se segue é baseado no documento File Format Description for Unsolvable
Boards for CSPLib escrito por Moshe Rubin (Mountain Vista Software) em dezembro de 2005.

### 4.1 Formato do input

As instâncias do problema Bimaru são constituídas por 3 partes:

1. A primeira linha é iniciada com a palavra ROW e contém informação relativa à contagem
    de posições ocupadas em cada linha da grelha.
2. A segunda linha é iniciada com a palavra COLUMN e contém informação relativa à conta-
    gem de posições ocupadas em cada coluna da grelha.
3. A terceira linha contém um inteiro que corresponde ao número de dicas.
4. As linhas seguintes são iniciadas com a palavra HINT e contêm as dicas correspondentes
    às posições pré-preenchidas.
Formalmente, cada uma das 4 partes acima descritas tem a seguinte formatação:
1. ROW \<count-0\> ... \<count-9\>
2. COLUMN <count-0> ... <count-9>
3. \<hint total\>
4. HINT \<row\> \<column\> \<hint value\>

Os valores possíveis para \<row\> e \<column\> são os números inteiros entre 0 e 9. O canto
superior esquerdo da grelha correponde às coordenadas (0,0).
Os valores possíveis para \<hint value\> são: W (water), C (circle), T (top), M (middle),
B (bottom), L (left) e R (right).

Exemplo

O ficheiro de input que descreve a instância da Figura 1 é o seguinte:

ROW 2 3 2 2 3 0 1 3 2 2
COLUMN 6 0 1 0 2 1 3 1 2 4
6
HINT 0 0 T
HINT 1 6 M
HINT 3 2 C
HINT 6 0 W
HINT 8 8 B
HINT 9 5 C

ROW\t2\t3\t2\t2\t3\t0\t1\t3\t2\t2\n
COLUMN\t6\t0\t1\t0\t2\t1\t3\t1\t2\t4\n
6\n
HINT\t0\t0\tT\n
HINT\t1\t6\tM\n
HINT\t3\t2\tC\n
HINT\t6\t0\tW\n
HINT\t8\t8\tB\n
HINT\t9\t5\tC\n

### 4.2 Formato do output

O output do programa deve descrever uma solução para o problema de Bimaru descrito no fi-
cheiro de input, i.e., uma grelha completamente preenchida que respeite as regras previamente
enunciadas. O output deve seguir o seguinte formato:

- 10 linhas, onde cada linha indica o conteúdo da respetiva linha da grelha.
- Nas posições pré-preenchidas (correspondentes a dicas) é colocada a respetiva letra
    maiúscula.
- Nas outras posições são colocadas as respetivas letras, mas minúsculas, com exceção
    das posições de água que, por questões de legibilidade, são representadas por um ponto.
- Todas as linhas, incluindo a última, são terminadas pelo carater newline, i.e.\n


Exemplo

O output que descreve a solução da Figura 2 é:

T.....t...
b.....M..t
......b..m
..C......m
c......c.b
..........
W...t.....
t...b...t.
m.......B.
b....C....

T.....t...\n
b.....M..t\n
......b..m\n
..C......m\n
c......c.b\n
..........\n
W...t.....\n
t...b...t.\n
m.......B.\n
b....C....\n