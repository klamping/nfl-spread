// This is an old file

const fs = require('fs');
const { smoothedAccuracyWithTotal } = require('./utils');

fs.readFile('results.json', 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading file:', err);
    return;
  }
  
  const jsonData = JSON.parse(data);
  const percentCorrect = calculatePercentCorrect(jsonData);
  
  fs.writeFile('percCorrect.json', JSON.stringify(percentCorrect, null, 2), (err) => {
    if (err) {
      console.error('Error writing file:', err);
    } else {
      console.log('Percent correct data saved to percCorrect.json');
    }
  });
});

const calculatePercentCorrect = (data) => {
  const spreadTotals = {};

  data.results.forEach(result => {
    Object.keys(result.spreadData).forEach(spread => {
      if (!spreadTotals[spread]) {
        spreadTotals[spread] = { total: 0, correct: 0 };
      }
      spreadTotals[spread].total += result.spreadData[spread].total;
      spreadTotals[spread].correct += result.spreadData[spread].correct;
    });
  });

  // Find the maximum total predictions for scaling
  const maxTotalPredictions = Math.max(...Object.values(spreadTotals).map(spread => spread.total));

  // Calculate total predictions to create weights
  let totalPredictions = 0;
  Object.keys(spreadTotals).forEach(spread => {
    totalPredictions += spreadTotals[spread].total;
  });

  // Calculate weighted percent correct
  const percentCorrect = {};
  Object.keys(spreadTotals).sort((a, b) => a - b).forEach(spread => {
    const totalPredictions = spreadTotals[spread].total;
    // Scale the spread total to factor in the spread with the most predictions
    const unweightedPercent = spreadTotals[spread].correct / totalPredictions;
    const adjustedWeight = smoothedAccuracyWithTotal(totalPredictions, unweightedPercent, maxTotalPredictions);
    percentCorrect[spread] = {
      raw: unweightedPercent,
      smoothed: adjustedWeight
    };
  });

  return percentCorrect;
};
