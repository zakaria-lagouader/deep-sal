import os
import numpy as np
from multiprocessing import Pool
from definitions import *
from configTrainSaliency01CNN import *

def process_mesh(file_name):
    # Load the 3D model
    mModel = loadObj(file_name)

    # Update geometry attributes
    updateGeometryAttibutes(
        mModel, 
        useGuided=useGuided, 
        numOfFacesForGuided=patchSizeGuided, 
        computeDeltas=False,
        computeAdjacency=False, 
        computeVertexNormals=False
    )

    train_data = []

    # Select patches
    patch_indices = [0, 19050]
    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in patch_indices]

    # Process each patch
    for i, patch in enumerate(patches):
        patch_faces = [mModel.faces[j] for j in patch]
        centroids = np.array([face.centroid for face in patch_faces])
        dists = np.linalg.norm(centroids - patch_faces[i].centroid, axis=1)
        dists /= np.max(dists)
        normals = np.array([face.faceNormal * dist for face, dist in zip(patch_faces, dists)])
        
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
        train_data.append((normals_reshaped + np.ones(normals_reshaped.shape)) / 2.0)
    
    # Save the training data
    np.save(f"./{os.path.basename(file_name).replace('.obj', '')}.npy", np.array(train_data, dtype=np.float32))
    print(f"Saved to cached/{os.path.basename(file_name).replace('.obj', '')}.npy")

if __name__ == "__main__":  
    process_mesh("data-1/bimba_decimated.obj")
