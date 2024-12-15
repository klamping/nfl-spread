const fs = require('fs');
const { parseArgs } = require('node:util');

const { values } = parseArgs({ 
  options: {
    date: {
      type: 'string',
      short: 'd',
    },
    location: {
      type: 'string',
      short: 'l'
    },
    show: {
      type: 'boolean',
      short: 's'
    },
    weight: {
        type: 'string',
        short: 'w'
    }
  }
});
const { date, location } = values;

// Load raw predictions data
const rawPredictionsFile = `./${location}/predictions/raw_predictions_${date}.json`;
const percCorrectFile = './percCorrect.json';
const confidencePercentsFile = './confidencePercent.json';

// Read and parse both files
const rawPredictions = JSON.parse(fs.readFileSync(rawPredictionsFile, 'utf8'));
const percCorrect = JSON.parse(fs.readFileSync(percCorrectFile, 'utf8'));
const confidencePercents = JSON.parse(fs.readFileSync(confidencePercentsFile, 'utf8'));

// Calculate confidence, weighted confidence, and sort by it
const predictionsWithConfidence = rawPredictions.map((item) => {
    const confidence = Math.abs(item.Prediction - 0.5);
    console.log('item.Prediction', item.Prediction);
    const confidencePercentInfo = confidencePercents.find(({ lowerRange, upperRange }) => item.Prediction > lowerRange && item.Prediction < upperRange);
    console.log('confidencePercentInfo', confidencePercentInfo)  
    const spreadPercent = percCorrect[item.Spread];
    const smoothedSpreadAccuracy = spreadPercent ? spreadPercent.smoothed : 0.45;
    const smoothedConfidenceAccuracy = confidencePercentInfo ? confidencePercentInfo.smoothedPercent : 0.45;
    const confidenceWeight = values.weight || 0.5;
    const spreadWeight = 1 - confidenceWeight;
    const weightedConfidence = confidenceWeight * smoothedConfidenceAccuracy + spreadWeight * smoothedSpreadAccuracy;
    const willCoverSpread = item.Prediction > 0.5;

    return {
        ...item,
        WillCoverSpread: willCoverSpread,
        WeightedConfidence: weightedConfidence,
        SpreadPercent: smoothedSpreadAccuracy,
        ConfidencePercent: smoothedConfidenceAccuracy,
        Confidence: confidence
    };
}).sort((a, b) => b.WeightedConfidence - a.WeightedConfidence);

// Assign ranks based on weighted confidence
const numPredictions = predictionsWithConfidence.length;
predictionsWithConfidence.forEach((item, index) => {
    item.points = numPredictions - index; // Most confident gets the highest rank
});

// Write the sorted and ranked data to a new file
const outputFilename = `./${location}/predictions/ranked_predictions_${date}.json`;
fs.writeFileSync(outputFilename, JSON.stringify(predictionsWithConfidence, null, 2), 'utf8');

// Log predictions using console.table
if (values.show) {
    console.log("\nRanked Predictions:\n");
    const prettyResults = predictionsWithConfidence.map(({ points, Matchup, WillCoverSpread, SpreadPercent, Prediction, Confidence, WeightedConfidence, ConfidencePercent }) => ({
        'Points': points,
        Matchup,
        'Will Cover Spread?': WillCoverSpread ? 'YES' : 'no',
        'Weighted Confidence': (WeightedConfidence * 100).toFixed(2),
        Prediction: Prediction.toFixed(4),
        Confidence: Confidence.toFixed(4),
        'Confidence % Correct': (ConfidencePercent* 100).toFixed(2) + '%',
        'Spread % Correct': (SpreadPercent* 100).toFixed(2) + '%',
    }));
    console.table(prettyResults);
}

// console.log(`\nRanked predictions have been saved to ${outputFilename}`);
