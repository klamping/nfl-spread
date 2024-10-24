const fs = require('fs');
const path = require('path');

// Directory containing JSON files
const directoryPath = './results/weekly-stats'; // Change this to your directory path

// Object to hold the collated data
const collatedData = {};

const useSingleWeek = true;
const week = '2024-10-16';
const outputPath = useSingleWeek ? week + '-team-stats' : "all-weekly-data-2";

// Read all files in the directory
fs.readdir(directoryPath, (err, files) => {
  if (err) {
    return console.error('Unable to scan directory:', err);
  }

  if (useSingleWeek) {
    files = files.filter((file) => file.includes(week));
  }

  // process.exit(0);
  // Loop through each file in the directory
  files.forEach((file) => {
    const filePath = path.join(directoryPath, file);

    // const week = '2024-10-09';

    // Ensure we're only reading .json files
    if (path.extname(file) === '.json') {
      // Read the content of the JSON file
      const fileContent = fs.readFileSync(filePath, 'utf8');

      try {
        // Parse the JSON content
        const jsonData = JSON.parse(fileContent);

        // Get the filename without the extension
        const fileNameWithoutExtension = path.basename(file, '.json');
        const fileNameRegex = /(\d{4}-\d{2}-\d{2})-([\w-]+)/;
        const isFileStats = fileNameWithoutExtension.match(fileNameRegex);
        if (isFileStats) {
          const filenameMatch = fileNameRegex.exec(fileNameWithoutExtension);

          const week = filenameMatch[1];
          const stat = filenameMatch[2];
          if (!collatedData[week]) {
            collatedData[week] = {};
          }
          for (const team of Object.keys(jsonData)) {
            if (!collatedData[week][team]) {
              collatedData[week][team] = {};
            }
            collatedData[week][team] = {
              ...collatedData[week][team],
              ...jsonData[team]
            };
          }
        } else {
          // Add the parsed data to the collatedData object
          collatedData[fileNameWithoutExtension] = jsonData;
        }
      } catch (error) {
        console.error(`Error parsing JSON file: ${filePath}`, error);
      }
    }
  });

  // Log the collated data or save it to a new file
  fs.writeFileSync(outputPath+'.json', JSON.stringify(collatedData, null, 2));
});
