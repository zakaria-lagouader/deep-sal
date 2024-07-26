from definitions import *
from configTrainSaliency01CNN import *
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numba
import traceback

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
    try:
        print(f"Processing {file_name}")
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

        train_data_array = np.asarray(train_data, dtype=np.float32)
        
        output_file = f"data-3/{os.path.basename(file_name).replace('.obj', '')}.npy"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, train_data_array)
        print(f"Saved to {output_file}")
        return f"Successfully processed and saved {file_name}"
    except Exception as e:
        return f"Error processing {file_name}: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    files = sorted(glob.glob("data-1/*.obj"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_mesh, files))
    
    for result in results:
        print(result)