import tensorflow as tf
from definitions import *
from configTrainSaliency01CNN import *
import trimesh

type = "continuous"
mesh_name = "data/skeleton_decimated.obj"
saliency_model = tf.keras.models.load_model("models/model-d-40.h5")
mModel = loadObj(mesh_name)
updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                        computeAdjacency=False, computeVertexNormals=False)
train_data = []

patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
# Rotation and train data formulation===============================================================================
for i, p in enumerate(patches):
    if i % 1000 == 0:
        print(f"{i /100}%")
    patchFacesOriginal = [mModel.faces[i] for i in p]
    positionsPatchFacesOriginal=np.asarray([pF.centroid for pF in patchFacesOriginal])
    normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
    vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
    vec = vec / np.linalg.norm(vec)
    axis, theta = computeRotation(vec, target)
    normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
    normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
    if reshapeFunction == "hilbert":
        for hci in range(np.shape(I2HC)[0]):
            normalsPatchFacesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchFacesOriginal[:,
                                                                        HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
    train_data.append((normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0)

train_data = np.asarray(train_data)
print(train_data.shape)
# train_data = train_data.reshape(train_data.shape[0], patchSide, patchSide, 3)
predictions = saliency_model.predict(train_data)

if type == "discrete":
    classes = np.linspace(0, 1, num_classes)
    predictions = classes[np.argmax(predictions, axis=1)]

mesh = trimesh.load(mesh_name)
mesh.visual.face_colors = trimesh.visual.interpolate(predictions, color_map='jet')
mesh.show()

