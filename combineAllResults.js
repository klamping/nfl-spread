const { writeFileSync } = require("fs");
const { smoothedAccuracyWithTotal } = require('./utils');

const dates = [
  "2024-09-25",
  "2024-10-02",
  "2024-10-09",
  "2024-10-16",
  "2024-10-23",
  "2024-10-30",
  "2024-11-06",
  "2024-11-13",
  "2024-11-20",
  "2024-11-27",
  "2024-12-03"
];

const predictions = dates.map(date => {
    return require(`./tsfour/predictions/prediction_results_${date}.json`);
});

const allPredictions = predictions.flat();

// group all the predictions by their spread
// Group by Spread and calculate counts
const groupedBySpread = allPredictions.reduce((acc, item) => {
    const spread = item.Spread;
  
    if (!acc[spread]) {
      acc[spread] = { count: 0, correctCount: 0 };
    }
  
    acc[spread].count++;
    if (item.correct) {
      acc[spread].correctCount++;
    }
  
    return acc;
  }, {});

  // Convert to array of objects for easier use
const groupedResults = Object.entries(groupedBySpread).map(([spread, stats]) => {
    const correctPercent = stats.correctCount / stats.count;
    return {
        Spread: spread,
        Count: stats.count,
        CorrectCount: stats.correctCount,
        raw: correctPercent,
        smoothed: smoothedAccuracyWithTotal(stats.count, correctPercent, allPredictions.length)
    };
});

const spreadPercents = groupedResults.reduce((acc, { Spread, raw, smoothed }) => {
    acc[Spread.toString()] = {
        raw,
        smoothed
    };
    return acc;
}, {})

const combinedPoints = predictions.map((weekPrediction, idx) => {
    const correctPucks = weekPrediction.filter(({ correct }) => correct);
    const numPoints = correctPucks.reduce((acc, { points }) => acc + points, 0);
    console.log(`Points earned in week ${dates[idx]}: ${numPoints}`);

    return {
        correctPucks: correctPucks.length,
        numPoints,
        allPucks: weekPrediction.length
    };
});
const cumulatedInfo = combinedPoints.reduce((acc, { correctPucks, numPoints, allPucks }) => {
    acc.correctPucks += correctPucks;
    acc.numPoints += numPoints;
    acc.allPucks += allPucks;
    return acc;
}, { correctPucks: 0, numPoints: 0, allPucks: 0 });
console.log('Total Correct:', cumulatedInfo.correctPucks);
console.log('Total Points:', cumulatedInfo.numPoints);
console.log('Total Pucks:', cumulatedInfo.allPucks);
console.log('Percent Correct:', ((cumulatedInfo.correctPucks / cumulatedInfo.allPucks) * 100).toFixed(2) + '%');

writeFileSync('./allResults.json', JSON.stringify(allPredictions), 'utf-8');
writeFileSync('./groupedResults.json', JSON.stringify(groupedResults), 'utf-8');
writeFileSync('./percCorrect.json', JSON.stringify(spreadPercents), 'utf-8');