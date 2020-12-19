import numpy as np
import math
import gco


def getdictid(lst):
    return str(lst[0]) + "," + str(lst[1])


# labels: [[u1, v1], [u2, v2], ...]
def graphcut(image, mask, labels):
    rows, cols = np.where(mask != 0)
    # sites: [[row1, col1], [row2, cole gg2], ...]
    sites = [[i, j] for (i, j) in zip(rows, cols)]
    n_sites = len(sites)
    n_labels = len(labels)
    sitesdict = {}
    for siteId in range(n_sites):
        sitesdict[getdictid(sites[siteId])] = siteId

    print("create graph")
    gc = gco.GCO()
    gc.create_general_graph(n_sites, n_labels)
    print("set data cost to exclude points in mask area")
    for siteId in range(n_sites):
        for labelId in range(n_labels):
            row = sites[siteId][0] + labels[labelId][0]
            col = sites[siteId][1] + labels[labelId][1]
            dataCost = 1000000
            if row < mask.shape[0] and col < mask.shape[1] and mask[row, col] == 0:
                dataCost = 0
            gc.set_site_data_cost(siteId, labelId, dataCost)

    print("set neighbors")
    s1, s2 = [], []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] == 0:
                continue
            #right
            if col < mask.shape[1] - 1 and getdictid([row, col + 1]) in sitesdict:
                s1.append(min(sitesdict[getdictid([row, col])], sitesdict[getdictid([row, col + 1])]))
                s2.append(max(sitesdict[getdictid([row, col])], sitesdict[getdictid([row, col + 1])]))
            #down
            if row < mask.shape[0] - 1 and getdictid([row + 1, col]) in sitesdict:
                s1.append(min(sitesdict[getdictid([row, col])], sitesdict[getdictid([row + 1, col])]))
                s2.append(max(sitesdict[getdictid([row, col])], sitesdict[getdictid([row + 1, col])]))
    ws = np.full(len(s1), 1)
    gc.set_all_neighbors(np.array(s1), np.array(s2), np.array(ws))

    print("set smooth cost")

    def smooth_cost_func(site1, site2, l1, l2):
        try:
            a1, b1 = sites[site1] + labels[l1], sites[site1] + labels[l2]
            a2, b2 = sites[site2] + labels[l1], sites[site2] + labels[l2]
            if a1[0] < image.shape[0] and a1[1] < image.shape[1] and b1[0] < image.shape[0] and b1[1] < image.shape[1] and a2[0] < image.shape[0] and a2[1] < image.shape[1] and b2[0] < image.shape[0] and b2[1] < image.shape[1]:
                n1 = np.linalg.norm(image[a1[0], a1[1]] - image[b1[0], b1[1]])
                n2 = np.linalg.norm(image[a2[0], a2[1]] - image[b2[0], b2[1]])
                #n1 = np.sum((image[a1[0], a1[1]] - image[b1[0], b1[1]])**2)
                #n2 = np.sum((image[a2[0], a2[1]] - image[b2[0], b2[1]])**2)
                return n1 + n2
            else:
                return 1000000
        except:
            print("exception caught in smooth_cost_func:", sites[site1] + labels[l1], sites[site1] + labels[l2],
                  sites[site2] + labels[l1], sites[site2] + labels[l2])
            print(image[a1])
            print(image[b1])
            print(image[a2])
            print(image[b2])
            return 1000000

    gc.set_smooth_cost_function(smooth_cost_func)

    print("swap")
    gc.swap(2)

    opti_labels = gc.get_labels()

    return sites, opti_labels
