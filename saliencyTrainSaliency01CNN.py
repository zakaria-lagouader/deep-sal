import tensorflow as tf
from definitions import *
from configTrainSaliency01CNN import *
from nets import CNNmodelKeras
import glob
import os
saliency_model=CNNmodelKeras(img_size,num_channels,num_classes,type)
train_data=[]
train_labels=[]
trainSet = sorted([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(rootdir + modelsDir + "*.obj")])[:30]
print(trainSet)
for modelName in trainSet:
    # ======Model information=====================================================================
    mModelSrc = rootdir + modelsDir + modelName + '.obj'
    print(modelName)
    if mode == "MESH":
        mModel = loadObj(mModelSrc)
        updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                                computeAdjacency=False, computeVertexNormals=False)
    if mode == "PC":
        mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
        V, inds = computePointCloudNormals(mModel, pointcloudnn)
        exportPLYPC(mModel, modelsDir + modelName + '_pcnorm_conf.ply')


    gtdata=np.loadtxt(rootdir + modelsDir + modelName +'.txt', delimiter=',')

    # #saliencyValue=saliencyValue/np.max(saliencyValue)
    print('Saliency ground truth data')
    if type == 'continuous':
        train_labels += gtdata.tolist()
    if type == 'discrete':
        train_labels += [int((num_classes - 1) * s) for s in gtdata.tolist()]


    if mode == "MESH":
        iLen = len(mModel.faces)
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

    if mode == "PC":
        iLen = len(mModel.vertices)
        patches = [neighboursByVertex(mModel, i, numOfElements)[0] for i in range(0, len(mModel.vertices))]
        # patches = np.random.choice(patches, numOfElements, replace=False)
        for i, p in enumerate(patches):
            if i % 1000 == 0:
                print(f"{i /100}%")
            patchVerticesOriginal = [mModel.vertices[i] for i in p]
            normalsPatchVerticesOriginal = np.asarray([pF.normal for pF in patchVerticesOriginal])
            vec = np.mean(np.asarray([fnm.normal for fnm in patchVerticesOriginal]), axis=0)
            vec = vec / np.linalg.norm(vec)
            axis, theta = computeRotation(vec, target)
            normalsPatchVerticesOriginal = rotatePatch(normalsPatchVerticesOriginal, axis, theta)
            normalsPatchVerticesOriginalR = normalsPatchVerticesOriginal.reshape((patchSide, patchSide, 3))
            if reshapeFunction == "hilbert":
                for hci in range(np.shape(I2HC)[0]):
                    normalsPatchVerticesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchVerticesOriginal[:,
                                                                                HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
            train_data.append((normalsPatchVerticesOriginalR + 1.0 * np.ones(np.shape(normalsPatchVerticesOriginalR))) / 2.0)

# Dataset and labels summarization ========================================================================
if type == 'continuous':
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    saliency_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])


if type == 'discrete':
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    saliency_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print(train_data.shape, train_labels.shape)

# load model
# saliency_model = tf.keras.models.load_model(rootdir+sessionsDir+'model-d-20.h5')

seppoint = int(0.8 * train_data.shape[0])
saliency_model.summary()
saliency_model_train = saliency_model.fit(x=train_data[:seppoint], y=train_labels[:seppoint], batch_size=batch_size, epochs=80, verbose=1)
saliency_model.save( rootdir+sessionsDir +'model-new-20.h5')

# Model evaluation ========================================================================
loss, acc = saliency_model.evaluate(train_data[seppoint:], train_labels[seppoint:], verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)