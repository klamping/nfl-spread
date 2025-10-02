const fs = require('fs');
const { ChartJSNodeCanvas } = require('chartjs-node-canvas');

// Configuration for the chart image
const width = 800; // Width of the image
const height = 600; // Height of the image
const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

// Read the JSON data from a file
const groupedResults = require('./groupedResults.json');

// Prepare data for the chart
const sortedResults = groupedResults.sort((a, b) => a.Spread - b.Spread);

const totalCounts = sortedResults.map(spread => spread.Count);
const correctCounts = sortedResults.map(spread => spread.CorrectCount);
const incorrectCounts = totalCounts.map((total, i) => total - correctCounts[i]);
const percentCorrect = sortedResults.map(spread => spread.raw * 100);
const percentCorrectSmoothed = sortedResults.map(spread => spread.smoothed * 100);

// Generate the chart
(async () => {
    const configuration = {
        type: 'bar',
        data: {
            labels: sortedResults.map(spread => spread.Spread),
            datasets: [
                {
                    label: 'Correct Count',
                    data: correctCounts,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    stack: 'stack1',
                },
                {
                    label: 'Incorrect Count',
                    data: incorrectCounts,
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                    stack: 'stack1',
                },
                {
                    label: 'Percent Correct',
                    data: percentCorrect,
                    type: 'line',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    fill: false,
                    yAxisID: 'y2',
                },
                {
                    label: 'Percent Correct (Smoothed)',
                    data: percentCorrectSmoothed,
                    type: 'line',
                    borderColor: 'rgba(99, 99, 132, 1)',
                    fill: false,
                    yAxisID: 'y2',
                },
            ],
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count',
                    },
                },
                y2: {
                    beginAtZero: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percent Correct (%)',
                    },
                },
            },
            plugins: {
                backgroundColor: 'rgba(255, 255, 255, 1)' // Adding a white background color
            },
        },
        plugins: [{
            id: 'background-colour',
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, chart.width, chart.height);
                ctx.restore();
            }
        }],
    };

    // Render the chart as an image
    const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration);

    // Save the image to a file
    const timestamp = Date.now();
    const imagePath = `./charts/chart-${timestamp}.png`;
    fs.writeFileSync(imagePath, imageBuffer);
    console.log(`Chart saved to ${imagePath}`);

    // Dynamically import the `open` module and open the image
    const open = (await import('open')).default;
    await open(imagePath);
})();
