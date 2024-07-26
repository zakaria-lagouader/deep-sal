from definitions import *
from configTrainSaliency01CNN import *
from multiprocessing import Pool
import os


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

    train_data = []
    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]

    # Process each patch
    for i, patch in enumerate(patches):
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
        train_data.append((normals_reshaped + 1) / 2)

    np.save(f"data-3/{os.path.basename(file_name).replace('.obj', '')}.npy", np.asarray(np.linalg.norm(train_data, axis=3), dtype=np.float32))
    print(f"saved to cached/{os.path.basename(file_name)}.npy")

if __name__ == "__main__":  
    with Pool(6) as p:
        p.map(process_mesh, sorted(glob.glob("data-1/*.obj"))[:2])
