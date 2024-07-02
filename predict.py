import tensorflow as tf
from definitions import *
from configTrainSaliency01CNN import *
import trimesh

type = "continuous"
mesh_name = "data-1/armadillo_decimated.obj"
saliency_model = tf.keras.models.load_model("models/saliency_model-f32-not-normalized.h5")
mModel = loadObj(mesh_name)
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
    # Vectorized Hilbert curve reshaping
    normalsPatchFacesOriginalR = np.zeros((patchSide, patchSide, 3))
    normalsPatchFacesOriginalR[I2HC[:, 0], I2HC[:, 1], :] = normalsPatchFacesOriginal.T[HC2I[I2HC[:, 0], I2HC[:, 1]], :]
    train_data.append((normalsPatchFacesOriginalR + 1.0) / 2.0)

train_data = np.asarray(train_data, dtype=np.float32)
# train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
predictions = saliency_model.predict(train_data)

if type == "discrete":
    classes = np.linspace(0, 1, num_classes)
    predictions = classes[np.argmax(predictions, axis=1)]

mesh = trimesh.load(mesh_name)
mesh.visual.face_colors = trimesh.visual.interpolate(predictions, color_map='jet')
mesh.show()

