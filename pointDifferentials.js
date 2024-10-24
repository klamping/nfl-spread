const stats = require("./all-weekly-data-2.json");
const season = require("./seasons.json");
const { writeFileSync } = require("fs");
const { generate } = require("csv-generate/sync");

const path = "results/pd-seasons.csv";

const skippedStat = "Red Zone Scoring Percentage (TD only) Last 1 Game";

// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";

const headings = [];
const firstWeek = Object.keys(stats)[0];
const firstTeam = Object.keys(stats[firstWeek])[0];
headings.push("Spread");
for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
    if (stat === skippedStat) {
      continue;  
    }
  headings.push(`Home: ${stat}`);
}
for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
    if (stat === skippedStat) {
      continue;  
    }
  headings.push(`Away: ${stat}`);
}
headings.push("Home Point Difference");
output += headings.join(",") + "\n";

for (const week of Object.keys(season)) {
  const games = season[week];

  //   "away": "Baltimore",
  //   "home": "Pittsburgh",
  //   "favorite": "Baltimore",
  //   "spread": 3,
  //   "score": {
  //     "away": 23,
  //     "home": 20
  //   },
  //   "is_favorite_home": 0,
  //   "pointDifferential": -3,
  //   "favoriteCovered": false
  for (const game of games) {
    if (!stats[week]) {
      console.log("No stats for week", week);
      continue;
    }
    const home = stats[week][game.home];
    const away = stats[week][game.away];

    const data = [];
    const isHomeFavorite = game.favorite === game.home;
    const spread = isHomeFavorite ? game.spread : -game.spread;
    data.push(spread);

    for (const [statName, stat] of Object.entries(home)) {
      if (statName === skippedStat) {
        continue;  
      }
      data.push(stat);
    }
    for (const [statName, stat] of Object.entries(away)) {
      if (statName === skippedStat) {
        continue;  
      }
      data.push(stat);
    }

    data.push(game.pointDifferential);

    output += data.join(",") + "\n";
  }
}

try {
  writeFileSync(path, output, "utf8");
  console.log("Data successfully saved to disk", path);
} catch (error) {
  console.log("An error has occurred ", error);
}