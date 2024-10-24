const stats = require("./all-weekly-data-2.json");
const season = require("./seasons.json");
const { writeFileSync } = require("fs");

const path = "results/seasons-again.csv";

// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";

const headings = [];
headings.push("Spread");
headings.push("Is Favorite Home Team");
const firstWeek = Object.keys(stats)[0];
const firstTeam = Object.keys(stats[firstWeek])[0];
for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
  headings.push(`Favorite: ${stat}`);
}
for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
  headings.push(`Underdog: ${stat}`);
}

// headings.push("Game Result: Favorite Points");
// headings.push("Game Result: Underdog Points");
headings.push("Favorite Covered Spread");
output += headings.join(",") + "\n";

for (const week of Object.keys(season)) {
  const games = season[week];
  for (const game of games) {
    if (!stats[week]) {
      console.log("No stats for week", week);
      continue;
    }
    const favorite = stats[week][game.favorite];
    const underdog = stats[week][game.underdog];

    const data = [];

    data.push(game.spread);
    data.push(game.is_favorite_home);
    for (const stat of Object.values(favorite)) {
      data.push(stat);
    }
    for (const stat of Object.values(underdog)) {
      data.push(stat);
    }

    // data.push(game.favoriteScore);
    // data.push(game.underdogScore);
    data.push(game.covered);

    output += data.join(",") + "\n";
  }
}

try {
  writeFileSync(path, output, "utf8");
  console.log("Data successfully saved to disk");
} catch (error) {
  console.log("An error has occurred ", error);
}