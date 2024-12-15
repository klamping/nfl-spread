const { writeFileSync } = require("fs");
const { smoothedAccuracyWithTotal } = require('./utils');

// Sample raw data (replace this with your actual dataset)
const predictions = require('./allResults.json');

const predictionCount = predictions.length;

// Function to group predictions and calculate percent correct
function groupPredictions(data, binSize = 0.01) {
    const bins = {};
    
    // Populate bins with data
    data.forEach(item => {
        const bin = Math.floor(item.Prediction / binSize) * binSize;
        if (!bins[bin]) {
            bins[bin] = { total: 0, correct: 0 };
        }
        bins[bin].total += 1;
        if (item.correct) {
            bins[bin].correct += 1;
        }
    });

    // Create a table with grouped predictions and percent correct
    const table = [];
    Object.keys(bins)
        .sort((a, b) => parseFloat(a) - parseFloat(b))
        .forEach(bin => {
            const binData = bins[bin];
            const percentCorrect = (binData.correct / binData.total) || 0;
            table.push({
                lowerRange: parseFloat(bin).toFixed(3) * 1,
                upperRange: (parseFloat(bin) + binSize).toFixed(3) * 1,
                PredictionRange: `${parseFloat(bin).toFixed(3)} - ${(parseFloat(bin) + binSize).toFixed(3)}`,
                PercentCorrect: percentCorrect.toFixed(4) * 1,
                smoothedPercent: smoothedAccuracyWithTotal(binData.total, percentCorrect, predictionCount),
                incorrect: binData.total - binData.correct,
                ...binData
            });
        });

    return table;
}

// Generate the table
const binSize = 0.05; // Group predictions by 0.01 increments
const resultTable = groupPredictions(predictions, binSize);

// Display the table in the console
console.table(resultTable);

// store the results as a JSON object
writeFileSync('./confidencePercent.json', JSON.stringify(resultTable), 'utf-8');
