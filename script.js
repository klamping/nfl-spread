const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// Load CSV data - nfl_games_data.csv
async function loadCSVData() {
    const csvFilePath = path.resolve(
      __dirname,
      "all-ATS-results-2023.csv"
    );
    const csvData = fs.readFileSync(csvFilePath, 'utf-8');

    const rows = csvData.split('\n').slice(1).map(row => row.split(',').map(Number));
    const features = rows.map(row => row.slice(0, -1)); // Features excluding the 'covered_spread'
    const labels = rows.map(row => row.slice(-1)[0]);   // Target (last column 'covered_spread')
    return { features, labels };
}

// Normalize the data
function normalizeData(features) {
    const dataTensor = tf.tensor2d(features);
    const { mean, variance } = tf.moments(dataTensor, 0);
    const normalizedData = dataTensor.sub(mean).div(variance.sqrt());
    return normalizedData;
}

// Build the TensorFlow.js Model
function buildModel(inputShape) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [inputShape],
        units: 64,
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));  // Binary classification (0/1)

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Train the model
async function trainModel(model, features, labels) {
    const inputTensor = tf.tensor2d(features);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    return model.fit(inputTensor, labelTensor, {
        epochs: 5000,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });
}

// Predict the outcome for upcoming games
async function predictOutcome(model, upcomingGamesData, mean, variance) {
    const normalizedData = tf.tensor2d(upcomingGamesData).sub(mean).div(variance.sqrt());
    const predictions = model.predict(normalizedData);

    predictions.array().then(result => {
        const predictedClasses = result.map(pred => pred > 0.5 ? 1 : 0);
        console.log('Predictions for upcoming games (0 = no cover, 1 = cover):');
        console.log(predictedClasses);
    });
}

// Main function to load data, train the model, and make predictions
(async function main() {
    const { features, labels } = await loadCSVData();

    // Normalize features
    const normalizedFeatures = normalizeData(features);

    // Build and train the model
    const model = buildModel(normalizedFeatures.shape[1]);
    await trainModel(model, normalizedFeatures.arraySync(), labels);

    // Predict upcoming games (replace with actual upcoming game data)
    const upcomingGames = [ // 0, 0, 0, 1, 1
        [3.5,1,-3.2,-17,-7,20.9,17.3,9,95.3,92,86,232.6,208,215,1.3,2,1,4.9,5,3,0.1,0,0,61.9,61,72,62,65,62,24.1,34.3,16,368.9,342,313,112.6,110,101,256.3,232,212,4.8,4,8,22.4,22,25,103.8,86.7,132,252.7,249,241,1.7,1.7,1,5.8,3.3,3,0.1,0,0,56.4,55.7,68,60.1,58,53,17.6,18,17,285.8,224.7,263,112.9,104.3,104,172.9,120.3,159],
        [5,0,1.4,0.3,-4,26.4,27,31,128.6,146.3,91,229.3,207.3,184,1.5,1.7,1,5.8,6,4,0.2,0.3,1,59.6,61.7,82,62.8,59.7,49,25.1,26.7,35,352.4,346,449,103.1,142.3,221,249.4,203.7,228,-9.9,-9,-1,14.9,18.7,25,110.3,90.3,105,161.3,201,284,1.1,0.7,1,5.4,5,7,0.3,0.7,1,61.2,62,66,61,56.3,67,24.8,27.7,26,365.6,384,391,132.4,120.7,105,233.3,263.3,286],
        [5.5,1,12.1,6.3,17,29.4,30.3,27,141.5,149.7,184,263.1,264.7,224,1.1,1.7,0,6,4.7,3,0.1,0.3,0,58,56,43,60.7,61.7,78,17.3,24,10,306.8,334.7,225,88.5,132.7,62,218.3,202,163,1.6,5.7,1,23.9,28,26,121,144.7,105,244.6,286.7,286,1.1,1.7,3,5.4,3.7,3,0,0,0,60.6,55.3,67,63.1,77,66,22.3,22.3,25,340.3,341.7,389,105.7,73,105,234.6,268.7,284],
        [13,0,10.4,-7.3,1,29.4,16.7,20,111.8,82.3,61,255.5,223.7,323,0.9,1.3,2,7.1,5.3,5,0.4,0,0,59.1,65.7,67,62.7,55.3,57,19.1,24,19,307.2,382,420,116.3,160.7,125,190.9,221.3,295,-10.1,-9,-17,19.9,19.3,10,96.4,81,62,224.7,174.7,163,1.8,2,2,5.2,5.3,5,0,0,0,64.1,81.7,78,60.4,45.3,43,30,28.3,27,385.8,411.3,408,126.5,181.3,184,259.3,230,224],
        [2.5,0,8.3,9.7,6,26.9,27.3,27,130.3,166,127,238.1,156.3,154,1.6,1.3,1,6.3,4,4,0.1,0.3,1,58.6,56.7,49,62.8,64,71,18.6,17.7,21,309.2,254,294,110.8,96.7,103,198.4,157.3,191,7,-1.7,-37,30.1,23.7,19,137.5,107.3,154,271.7,239.3,221,1.4,1,3,5.8,6.3,5,0.3,0,0,59.6,52.7,55,60.4,63.7,64,23.1,25.3,56,308.6,311,491,95.1,93.3,160,213.5,217.7,331]
    ];

    // Using the same normalization process on the new data
    const { mean, variance } = tf.moments(normalizedFeatures, 0);
    await predictOutcome(model, upcomingGames, mean, variance);
})();
