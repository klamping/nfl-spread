const fs = require('fs');
const path = require('path');

// Directory containing JSON files
const directoryPath = './results/season-matchups';

// Object to hold the collated data
let collatedData = {};

const outputPath = "seasons";

// Read all files in the directory
fs.readdir(directoryPath, (err, files) => {
  if (err) {
    return console.error('Unable to scan directory:', err);
  }

  // Loop through each file in the directory
  files.forEach((file) => {
    const filePath = path.join(directoryPath, file);

    // Ensure we're only reading .json files
    if (path.extname(file) === '.json') {
      // Read the content of the JSON file
      const fileContent = fs.readFileSync(filePath, 'utf8');

      try {
        // Parse the JSON content
        const jsonData = JSON.parse(fileContent);

        // Add the parsed data to the collatedData object
        collatedData = {
          ...collatedData,
          ...jsonData
        };
      } catch (error) {
        console.error(`Error parsing JSON file: ${filePath}`, error);
      }
    }
  });

  // Log the collated data or save it to a new file
  fs.writeFileSync(outputPath+'.json', JSON.stringify(collatedData, null, 2));
});
