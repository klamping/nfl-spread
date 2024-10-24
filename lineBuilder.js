const stats = require("./2024-10-02.json");
const games = require('./4-seasons.json');

const headings = [];
for (const stat of Object.keys(stats['2024-09-04']['Detroit'])) {
    headings.push(`Home: ${stat}`);
}
for (const stat of Object.keys(stats["2024-09-04"]["Detroit"])) {
  headings.push(`Away: ${stat}`);
}
console.log(headings.join(','));

for (const game of games) {
    const favorite = stats[game.favorite];
    const underdog = stats[game.underdog];

    const data = [];

    data.push(game.spread);
    data.push(game.is_favorite_home);
    for (const stat of Object.values(favorite)) {
      data.push(stat);
    }
    for (const stat of Object.values(underdog)) {
      data.push(stat);
    }
    data.push(game.covered);

    console.log(data.join(','));
}