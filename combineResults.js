const fs = require('fs');
const path = require('path');
const { parseArgs } = require('node:util');

const { values } = parseArgs({ 
  options: {
    date: {
      type: 'string',
      short: 'd',
    }
  }
});
const { date } = values;

function calculateCombinedPoints(coverFilePath, pointDiffFilePath) {
    // Load the CSV files
    const csv = require('csv-parser');
    const coverData = [];
    const pointDiffData = [];

    return new Promise((resolve, reject) => {
        // Read the cover file
        fs.createReadStream(coverFilePath)
            .pipe(csv())
            .on('data', (row) => {
                coverData.push(row);
            })
            .on('end', () => {
                // Read the point difference file
                fs.createReadStream(pointDiffFilePath)
                    .pipe(csv())
                    .on('data', (row) => {
                        pointDiffData.push(row);
                    })
                    .on('end', () => {
                        // Merge the data on the 'Matchup' column
                        const combinedPoints = [];
                        const mergedData = coverData.map((coverRow) => {
                            const pointDiffRow = pointDiffData.find(
                                (pdRow) => pdRow['Matchup'] === coverRow['Matchup']
                            );

                            if (pointDiffRow) {
                                let combinedPointValue;

                                const bothAgree = coverRow['Favorite Covers'] === pointDiffRow['Favorite Covers'];

                                if (bothAgree) {
                                    // If the prediction matches, use a higher weighted average of the points assigned from both sets
                                    combinedPointValue =
                                        0.75 * parseInt(coverRow['Points Assigned']) +
                                        0.75 * parseInt(pointDiffRow['Points Assigned']);
                                } else {
                                    // If the prediction doesn't match, prioritize based on the average points assigned, but with lower weight
                                    combinedPointValue =
                                        0.25 * parseInt(coverRow['Points Assigned']) +
                                        0.25 * parseInt(pointDiffRow['Points Assigned']);
                                }

                                combinedPoints.push({
                                    Matchup: coverRow['Matchup'],
                                    'Favorite Covers (Cover File)': coverRow['Favorite Covers'],
                                    'Favorite Covers (Point Diff File)': pointDiffRow['Favorite Covers'],
                                    'Both Agree?': bothAgree ? 'YES' : 'no',
                                    'Points Assigned (Cover File)': coverRow['Points Assigned'],
                                    'Points Assigned (Point Diff File)': pointDiffRow['Points Assigned'],
                                    'Combined Points': combinedPointValue,
                                });
                            }
                        });

                        // Sort the results to prioritize where both sets agree, then prioritize disagreements by closest agreement
                        combinedPoints.sort((a, b) => {
                            if (a['Both Agree?'] === 'YES' && b['Both Agree?'] !== 'YES') {
                                return -1;
                            } else if (a['Both Agree?'] !== 'YES' && b['Both Agree?'] === 'YES') {
                                return 1;
                            } else if (a['Both Agree?'] === 'no' && b['Both Agree?'] === 'no') {
                                // Sort disagreements by the sum of points assigned (closest agreement)
                                const aPointsSum =
                                    parseInt(a['Points Assigned (Cover File)']) +
                                    parseInt(a['Points Assigned (Point Diff File)']);
                                const bPointsSum =
                                    parseInt(b['Points Assigned (Cover File)']) +
                                    parseInt(b['Points Assigned (Point Diff File)']);
                                return aPointsSum - bPointsSum;
                            } else {
                                return b['Combined Points'] - a['Combined Points'];
                            }
                        });

                        // Assign new 'Points Assigned' values from 1 to number of rows based on sorted order
                        combinedPoints.forEach((item, index) => {
                            item['New Points Assigned'] = combinedPoints.length - index;
                        });

                        // Create a new dataset for the CSV output
                        const outputData = combinedPoints.map((item) => {
                            const favoriteCovers = item['Favorite Covers (Cover File)'] === 'True' ? 'True' : 'False';
                            return {
                                Matchup: item['Matchup'],
                                'Favorite Covers': favoriteCovers,
                                'Points Assigned': item['New Points Assigned'],
                            };
                        });

                        // Write the output to a new CSV file
                        const outputFilePath = path.join(__dirname, 'tsthree', 'predictions', `combined_points_${date}.csv`);
                        const writeStream = fs.createWriteStream(outputFilePath);
                        writeStream.write('Matchup,Favorite Covers,Points Assigned\n');
                        outputData.forEach((row) => {
                            writeStream.write(`${row['Matchup']},${row['Favorite Covers']},${row['Points Assigned']}\n`);
                        });
                        writeStream.end();

                        resolve(combinedPoints);
                    })
                    .on('error', (error) => reject(error));
            })
            .on('error', (error) => reject(error));
    });
}

// Example usage
const coverFilePath = path.join(__dirname, 'tsthree', 'predictions', `predictions_cover_${date}.csv`);
const pointDiffFilePath = path.join(__dirname, 'tsthree', 'predictions', `predictions_point_diff_${date}.csv`);

calculateCombinedPoints(coverFilePath, pointDiffFilePath)
    .then((combinedData) => {
        console.table(combinedData); // Display the combined data
    })
    .catch((error) => {
        console.error('Error:', error);
    });
