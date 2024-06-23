import tensorflow as tf
from definitions import *
from configTrainSaliency01CNN import *
from nets import CNNmodelKeras
import glob
import os
import numpy as np
from multiprocessing import Pool

# Function to process a single model
def process_model(modelName):
    train_data = []
    train_labels = []

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

    gtdata = np.loadtxt(rootdir + modelsDir + modelName + '.txt', delimiter=',')

    print('Saliency ground truth data')
    if type == 'continuous':
        train_labels += gtdata.tolist()
    if type == 'discrete':
        train_labels += [int((num_classes - 1) * s) for s in gtdata.tolist()]

    if mode == "MESH":
        patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
        for i, p in enumerate(patches):
            if i % 1000 == 0:
                print(f"{i / 100}%")
            patchFacesOriginal = [mModel.faces[i] for i in p]
            # positionsPatchFacesOriginal = np.asarray([pF.centroid for pF in patchFacesOriginal])
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
        for i, p in enumerate(patches):
            if i % 1000 == 0:
                print(f"{i / 100}%")
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

    return train_data, train_labels

# Main code
if __name__ == "__main__":
    # load model
    saliency_model = tf.keras.models.load_model(rootdir+sessionsDir+'model-new-20.h5')
    # saliency_model = CNNmodelKeras(img_size, num_channels, num_classes, type)
    trainSet = sorted([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(rootdir + modelsDir + "*.obj")])[20:]
    print(trainSet)

    # Use multiprocessing to process models in parallel
    with Pool() as pool:
        results = pool.map(process_model, trainSet)

    # Collect results from the pool
    train_data = []
    train_labels = []
    for data, labels in results:
        train_data.extend(data)
        train_labels.extend(labels)

    # Convert train_data and train_labels to numpy arrays
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)

    # Dataset and labels summarization
    if type == 'continuous':
        saliency_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    if type == 'discrete':
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        saliency_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(train_data.shape, train_labels.shape)

    # Train the model
    seppoint = int(0.8 * train_data.shape[0])
    saliency_model.summary()
    saliency_model_train = saliency_model.fit(x=train_data[:seppoint], y=train_labels[:seppoint], batch_size=batch_size, epochs=120, verbose=1)
    saliency_model.save(rootdir + sessionsDir + 'model-new-20.h5')

    # Model evaluation
    loss, acc = saliency_model.evaluate(train_data[seppoint:], train_labels[seppoint:], verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
