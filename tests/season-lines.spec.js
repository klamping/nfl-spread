const { test } = require('@playwright/test');
const mapTeamName = require('../mapTeamName.js');
const { writeFileSync } = require("fs");

const { weeks } = require('./weeks.json');
const seasonsPath = `./results/10-seasons.json`;

// const weeks = ["2024-10-16"];
// const seasonsPath = `results/${weeks[0]}-week.json`;

// taking a list of weeks, get the years we're going to check
const weeksByYear = weeks.reduce((acc, week) => {
  const date = week.split("-");
  let year = date[0];
  const month = date[1];

  if (month === "01") {
    year = year - 1;
  }

  if (!acc[year]) {
    acc[year] = [];
  }

  acc[year].push(week);
  return acc;
}, {});

test.describe("Get season results", async () => {
  const seasonResults = {};
  
  for (const [year, weeks] of Object.entries(weeksByYear)) {
    test("Get season results for year " + year, async ({ page }) => {
      await page.goto(
        `https://www.sportsoddshistory.com/nfl-game-season/?y=` + year,
        {
          waitUntil: "domcontentloaded",
          timeout: 10000,
        }
      );

      for (const [index, week] of weeks.entries()) {
        // we start at week 3, so we need to add 3 to the index, plus one because the id is 1-indexed
        const weekGames = await page
          .locator(`a[id="${index + 4}"]~table`)
          .first()
          .locator("tbody:nth-child(2) tr")
          .all();

        const results = [];

        for (const game of weekGames) {
          // 0 Day, 1 Date, 2 Time, 3 is home, 4 Favorite, 5 Score, 6 Spread, 7 is away, 8 Underdog
          const isHome = await game.locator("td").nth(3).textContent();
          const favorite = mapTeamName(
            await game.locator("td").nth(4).textContent()
          );
          const spreadDetail = await game.locator("td").nth(6).textContent();
          const underdog = mapTeamName(
            await game.locator("td").nth(8).textContent()
          );

          // split the spread
          let covered, spread;
          if (spreadDetail === "W PK") {
            covered = 1;
            spread = 0;
          } else if (spreadDetail === "L PK") {
            covered = 0;
            spread = 0;
          } else {
            const spreadRegex = /([WLP]) -(\d+(?:\.\d)?)/g;
            const spreadMatch = spreadRegex.exec(spreadDetail);
            try {
              covered = spreadMatch[1] === "W" ? 1 : 0;
              spread = parseFloat(spreadMatch[2]);
            } catch (error) {
              throw new Error(
                `Error parsing spread ${spreadDetail} ${spreadMatch}`
              );
            }
          }
          const score = await game.locator("td").nth(5).textContent();
          const scoreRegex = /[WLT] (\d+)-(\d+)/g;
          const scoreMatch = scoreRegex.exec(score);
          const favoriteScore = scoreMatch[1];
          const underdogScore = scoreMatch[2];
          const scoreDifferential = favoriteScore - underdogScore;

          results.push({
            favorite,
            underdog,
            is_favorite_home: isHome === "@" ? 1 : 0,
            spread,
            covered,
            favoriteScore,
            underdogScore,
            scoreDifferential,
          });
        }

        seasonResults[week] = results;
      }
    });
  }

  test.afterAll(() => {
    try {
      writeFileSync(seasonsPath, JSON.stringify(seasonResults, null, 2), "utf8");
      console.log("Data successfully saved to disk");
    } catch (error) {
      console.log("An error has occurred ", error);
    }
  });
});