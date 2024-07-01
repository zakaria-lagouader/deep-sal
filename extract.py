from definitions import *
from configTrainSaliency01CNN import *
from multiprocessing import Pool
import os


def process_mesh(file_name):
    mModel = loadObj(file_name)

    updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                            computeAdjacency=False, computeVertexNormals=False)
    train_data = []
    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
    # Rotation and train data formulation===============================================================================
    for i, p in enumerate(patches):
        patchFacesOriginal = [mModel.faces[i] for i in p]
        normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
        vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
        vec = vec / np.linalg.norm(vec)
        axis, theta = computeRotation(vec, target)
        normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
        normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
        for hci in range(np.shape(I2HC)[0]):
            normalsPatchFacesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchFacesOriginal[:,
                                                                        HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
        train_data.append((normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0)

    np.save(f"cached/{os.path.basename(file_name).replace('.obj', '')}.npy", np.asarray(train_data, dtype=np.float32))
    print(f"saved to cached/{os.path.basename(file_name)}.npy")

if __name__ == "__main__":  
    with Pool(8) as p:
        p.map(process_mesh, sorted(glob.glob(f"data-1/*.obj")))
