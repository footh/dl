import numpy as np

valid_zones_by_slice = [
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    [False, False, True, True, False, False, True, False, False, True, False, True, False, True, False, True, False], #TODO: maybe 5, 9, 11, 13, 15 belongs
    [False, False, True, True, False, False, True, False, False, True, False, True, False, True, False, True, False],
    [False, False, True, True, False, False, True, False, False, True, False, True, False, True, False, True, False], #TODO: maybe 9, 11, 13, 15, 17 belongs
    [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True], #TODO: maybe ditch 3, 4, make decision based on sampling
    [True, True, False, False, False, True, False, True, False, False, True, False, True, False, True, False, False], #TODO: maybe 9, 12, 14, 16, 17 belong
    [True, True, False, False, False, True, False, True, False, False, True, False, True, False, True, False, False],
    [True, True, False, False, False, True, False, True, False, False, True, False, True, False, True, False, False], #TODO: maybe 5, 9, 12, 14, 16 belong
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]       
]

zone_padding = np.zeros((16, 18, 4), dtype=np.int16)

zone_padding[0][1] = zone_padding[1][1] = zone_padding[2][1] = [-90,0,0,10]
zone_padding[6][1] = zone_padding[7][1] = zone_padding[8][1] = [0,0,90,10]
zone_padding[9][1] = zone_padding[10][1] = [0,0,90,10]
zone_padding[11][1] = zone_padding[12][1] = zone_padding[13][1] = [-50,0,50,0]
zone_padding[14][1] = zone_padding[15][1] = [-90,0,0,10]

zone_padding[0][3] = zone_padding[1][3] = zone_padding[2][3] = [0,0,90,10]
zone_padding[3][3] = zone_padding[4][3] = zone_padding[5][3] = [-50,0,50,0]
zone_padding[6][3] = zone_padding[7][3] = zone_padding[8][3] = [-90,0,0,10]
zone_padding[9][3] = zone_padding[10][3] = [-90,0,0,10]
zone_padding[14][3] = zone_padding[15][3] = [0,0,90,10]

zone_padding[0][5] = zone_padding[1][5] = zone_padding[2][5] = zone_padding[14][5] = zone_padding[15][5] = [0,0,0,10]
zone_padding[0][6] = zone_padding[1][6] = zone_padding[2][6] = zone_padding[14][6] = zone_padding[15][6] = [-50,0,10,10]
zone_padding[0][7] = zone_padding[1][7] = zone_padding[2][7] = zone_padding[14][7] = zone_padding[15][7] = [-10,0,50,10]
zone_padding[0][8] = zone_padding[1][8] = zone_padding[2][8] = zone_padding[14][8] = zone_padding[15][8] = [-50,0,10,10] #slice 2 maybe ignore 8 
zone_padding[0][9] = zone_padding[1][9] = zone_padding[2][9] = zone_padding[14][9] = zone_padding[15][9] = [-10,0,10,10] #slice 2 maybe not expand 9
zone_padding[0][10] = zone_padding[1][10] = zone_padding[2][10] = zone_padding[14][10] = zone_padding[15][10] = [-10,0,50,10]

zone_padding[0][11] = zone_padding[0][13] = zone_padding[0][15] = zone_padding[1][11] = zone_padding[1][13] = zone_padding[1][15] = [-50,0,0,0]
zone_padding[2][11] = zone_padding[2][13] = zone_padding[2][15] = [-25,0,25,0]
zone_padding[6][11] = zone_padding[6][13] = zone_padding[6][15] = [0,0,50,0]
zone_padding[7][11] = zone_padding[7][13] = zone_padding[7][15] = [20,0,50,0]
zone_padding[8][11] = zone_padding[8][13] = zone_padding[8][15] = zone_padding[9][11] = zone_padding[9][13] = zone_padding[9][15] = [0,0,50,0]
zone_padding[10][11] = zone_padding[10][13] = zone_padding[10][15] = [-25,0,25,0]
zone_padding[11][11] = zone_padding[11][13] = zone_padding[11][15] = zone_padding[12][11] = zone_padding[12][13] = zone_padding[12][15] = zone_padding[13][11] = zone_padding[13][13] = zone_padding[13][15] = [-50,0,0,0]
zone_padding[14][11] = zone_padding[14][13] = zone_padding[14][15] = zone_padding[15][11] = zone_padding[15][13] = zone_padding[15][15] = [-50,0,-25,0]

zone_padding[0][12] = zone_padding[0][14] = zone_padding[0][16] = zone_padding[1][12] = zone_padding[1][14] = zone_padding[1][16] = [0,0,50,0]
zone_padding[2][12] = zone_padding[2][14] = zone_padding[2][16] = [-15,0,30,0]
zone_padding[3][12] = zone_padding[3][14] = zone_padding[3][16] = zone_padding[4][12] = zone_padding[4][14] = zone_padding[4][16] = zone_padding[5][12] = zone_padding[5][14] = zone_padding[5][16] = [0,0,50,0]
zone_padding[6][12] = zone_padding[6][14] = zone_padding[6][16] = [-50,0,5,0]
zone_padding[7][12] = zone_padding[7][14] = zone_padding[7][16] = [-50,0,-15,0]
zone_padding[8][12] = zone_padding[8][14] = zone_padding[8][16] = zone_padding[9][12] = zone_padding[9][14] = zone_padding[9][16] = [-50,0,0,0]
zone_padding[10][12] = zone_padding[10][14] = zone_padding[10][16] = [-25,0,15,0]
zone_padding[14][12] = zone_padding[14][14] = zone_padding[14][16] = zone_padding[15][12] = zone_padding[15][14] = zone_padding[15][16] = [-20,0,30,0]

zone_padding[3][7] = zone_padding[3][10] = zone_padding[11][6] = zone_padding[11][8] = [-50,0,50,10]
zone_padding[4][7] = zone_padding[4][10] = zone_padding[12][6] = zone_padding[12][8] = [-50,0,50,10]
zone_padding[5][7] = zone_padding[5][10] = zone_padding[13][6] = zone_padding[13][8] = [-50,0,50,10]

zone_padding[6][17] = zone_padding[7][17] = zone_padding[8][17] = zone_padding[9][17] = zone_padding[10][17] = [0,0,0,10]
zone_padding[6][7] = zone_padding[7][7] = zone_padding[8][7] = zone_padding[9][7] = zone_padding[10][7] = [-50,0,10,10]
zone_padding[6][6] = zone_padding[7][6] = zone_padding[8][6] = zone_padding[9][6] = zone_padding[10][6] = [-10,0,50,10]
zone_padding[6][10] = zone_padding[7][10] = zone_padding[8][10] = zone_padding[9][10] = zone_padding[10][10] = [-50,0,10,10]
zone_padding[6][9] = zone_padding[7][9] = zone_padding[8][9] = zone_padding[9][9] = zone_padding[10][9] = [-10,0,10,10]
zone_padding[6][8] = zone_padding[7][8] = zone_padding[8][8] = zone_padding[9][8] = zone_padding[10][8] = [-10,0,50,10]

def apply_padding(zones, max_width=512, max_height=660):
    for i in range(zones.shape[0]):
        for j in range(1, zones.shape[1]):
            if np.sum(zones[i][j]) > 0:
                update = zones[i][j] + zone_padding[i][j]
                update = [max(0, update[0]), max(0, update[1]), min(max_width, update[2]), min(max_height, update[3])]
                zones[i][j] = update
