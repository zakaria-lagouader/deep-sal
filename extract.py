from definitions import *
from configTrainSaliency01CNN import *
from multiprocessing import Pool
import os
import glob
import numpy as np

def process_mesh(file_name):
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

    num_faces = len(mModel.faces)
    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(num_faces)]
    train_data = np.empty((num_faces, patchSide, patchSide, 3), dtype=np.float32)

    for idx, patch in enumerate(patches):
        patch_faces = [mModel.faces[j] for j in patch]
        normals = np.array([face.faceNormal for face in patch_faces])
        
        # Calculate rotation vector
        vec = np.mean([face.area * face.faceNormal for face in patch_faces], axis=0)
        vec /= np.linalg.norm(vec)
        axis, theta = computeRotation(vec, target)
        normals = rotatePatch(normals, axis, theta)
        
        # Reshape normals
        normals_reshaped = normals.reshape((patchSide, patchSide, 3))
        
        # Apply I2HC and HC2I transformations
        for hci in range(I2HC.shape[0]):
            i, j = I2HC[hci]
            normals_reshaped[i, j, :] = normals[:, HC2I[i, j]]
        
        # Normalize the data
        train_data[idx] = (normals_reshaped + 1) / 2

    # Save the result
    output_file = f"data-3/{os.path.basename(file_name).replace('.obj', '')}.npy"
    np.save(output_file, np.linalg.norm(train_data, axis=3))
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    obj_files = sorted(glob.glob("data-1/*.obj"))
    with Pool(8) as p:
        p.map(process_mesh, obj_files)