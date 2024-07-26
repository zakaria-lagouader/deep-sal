from definitions import *
from configTrainSaliency01CNN import *
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numba

@numba.jit(nopython=True)
def process_patch(patch_faces, I2HC, HC2I):
    normals = np.array([face.faceNormal for face in patch_faces])
    
    vec = np.mean([face.area * face.faceNormal for face in patch_faces], axis=0)
    vec /= np.linalg.norm(vec)
    axis, theta = computeRotation(vec, target)
    normals = rotatePatch(normals, axis, theta)
    
    normals_reshaped = normals.reshape((patchSide, patchSide, 3))
    
    for hci in range(I2HC.shape[0]):
        i, j = I2HC[hci]
        normals_reshaped[i, j, :] = normals[:, HC2I[i, j]]
    
    return np.linalg.norm((normals_reshaped + 1) / 2, axis=2)

def process_mesh(file_name):
    print(f"processing {file_name}")
    mModel = loadObj(file_name)

    updateGeometryAttibutes(
        mModel, 
        useGuided=useGuided, 
        numOfFacesForGuided=patchSizeGuided, 
        computeDeltas=False,
        computeAdjacency=False, 
        computeVertexNormals=False
    )

    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(len(mModel.faces))]

    train_data = []
    for patch in patches:
        patch_faces = [mModel.faces[j] for j in patch]
        train_data.append(process_patch(patch_faces, I2HC, HC2I))

    output_file = f"data-3/{os.path.basename(file_name).replace('.obj', '')}.npy"
    np.save(output_file, np.asarray(train_data, dtype=np.float32))
    print(f"saved to {output_file}")

if __name__ == "__main__":
    files = sorted(glob.glob("data-1/*.obj"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_mesh, files)