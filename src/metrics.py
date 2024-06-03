import numpy as np 


def dice(vol1, vol2, labels):
    """ calculate dice score between vol1 and vol2 """
    diceList = np.zeros(len(labels))
    idx = 0 
    for label in labels:
        labelList = labels[label]
        labelLen = len(labelList)
        labelSum = 0 
        for labelOne in labelList:
            top = 2 * np.sum(np.logical_and(vol1 == labelOne, vol2 == labelOne))
            bottom = np.sum(vol1 == labelOne) + np.sum(vol2 == labelOne)
            bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon to avoid divided by 0
            labelSum += top / bottom
        labelMean = labelSum / labelLen 
        diceList[idx] = labelMean
        idx += 1 
    return diceList



# def jacobian_determinant(disp):
#     """
#     jacobian determinant of a displacement field (flow).
#     NB: to compute the spatial gradients, we use np.gradient.

#     Parameters:
#         disp: 2D or 3D displacement field of size [nb_channels, *vol_shape], 
#               where vol_shape is of len nb_channels

#     Returns:
#         jacobian determinant (scalar)
#     """
    
#     # check inputs
#     vol_shape = disp.shape[1:]
#     nb_channels = len(vol_shape)
#     assert nb_channels in (2, 3), 'flow has to be 2D or 3D'

#     # compute grid
#     grid_lst = nd.volsize2ndgrid(vol_shape)
#     grid = np.stack(grid_lst, axis=0)

#     # compute gradients
#     J = np.gradient(disp + grid)

#     # 3D glow
#     if nb_channels == 3:
#         dx = J[0]
#         dy = J[1]
#         dz = J[2]

#         # compute jacobian components
#         Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
#         Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
#         Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

#         return Jdet0 - Jdet1 + Jdet2

#     else: # must be 2 
        
#         dfdx = J[0]
#         dfdy = J[1] 
        
#         return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]