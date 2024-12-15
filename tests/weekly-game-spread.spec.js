const { test } = require("@playwright/test");
const { writeFileSync } = require("fs");
const mapTeamName = require("../mapTeamName.js");
const { smoothedAccuracy } = require('../utils.js');

const useSpecificWeek = false;
const weekNo = 14;
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
]
const date = dates[weekNo - 4];

let { weeks } = require("./weeks.json");

if (useSpecificWeek) {
  weeks = [ date ];
}

// To add... 
// - Travel and Rest Factors
// - Weather
// - Opponent Strength/SOS
// - Injuries
// - ATS Record

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

function convertTo24Hour(time) {
  const [timePart, modifier] = time.split(' ');
  let [hours, minutes] = timePart.split(':').map(Number);

  if (modifier === 'PM' && hours !== 12) {
      hours += 12;
  } else if (modifier === 'AM' && hours === 12) {
      hours = 0;
  }

  // Pad hours and minutes with leading zeros if needed and concatenate
  return `${hours.toString().padStart(2, '0')}${minutes.toString().padStart(2, '0')}`;
}

function calculateWinningPercentage(record) {
  // Remove parentheses and split the record into wins, losses, and draws
  const [wins, losses, draws] = record.slice(1, -1).split('-').map(Number);

  // Calculate the total number of games played
  const totalGames = wins + losses + draws;

  // Calculate the winning percentage
  const winningPercentage = totalGames === 0 ? 0 : (wins + (draws * 0.5)) / totalGames;

  // Return the winning percentage as a decimal or percentage string
  return {
    actual: winningPercentage.toFixed(3), // Example: returns "0.750" for (3-1-0)
    smoothed: smoothedAccuracy(totalGames, winningPercentage)
  };
}

for (const [year, weeks] of Object.entries(weeksByYear)) {
  test("Get weekly games for " + year, async ({ page }) => {
    const weekResults = {};
    for (const [index, week] of weeks.entries()) {
      const nflWeek = useSpecificWeek ? weekNo : (index + 4);
      await page.goto(
        `https://gridirongames.com/nfl-weekly-schedule/?Year=${year}&Week=${
          nflWeek
        }`,
          {
              waitUntil: "domcontentloaded"
          }
      );
      const matchupsTable = await page.locator(".nfl-matchups");
      let dayOfWeek;

      const allRows = await matchupsTable
        .locator('tbody tr')
        .all()

      const games = [];

      // go through each row, see if it's a date row, or a game row
      for (const row of allRows) {
        // if it's a date row, update the day of week
        const th = await row.locator('th');
        if (await th.count() > 0) {
          const dateText = await th.textContent();
          if (dateText.startsWith('Teams on bye')) {
            continue;
          }
          const weekday = dateText.split(',')[0];
          switch (weekday) {
            case 'Thursday':
              dayOfWeek = 0;
              break;
            case 'Friday':
              dayOfWeek = 1;
              break;
            case 'Saturday':
              dayOfWeek = 2;
              break;
            case 'Monday':
              dayOfWeek = 4;
              break;
            case 'Sunday':
              default:
              dayOfWeek = 3;
              break;
          }
        } else {
          const timeText = await row
            .locator("td:nth-child(1) > div:nth-child(2)")
            .textContent();
          const time = convertTo24Hour(timeText);

          const awayTeamInfo = await row
            .locator("td:nth-child(2) > div:nth-child(2)");
          const awayText = await awayTeamInfo.locator("> div:nth-child(1)")
            .textContent();
          const away = mapTeamName(awayText);

          const awayRecordText = await awayTeamInfo.locator("> div:nth-child(2)")
            .textContent();
            console.log(awayRecordText)
          const { actual: awayRecord, smoothed: smoothedAwayRecord } = calculateWinningPercentage(awayRecordText);
          console.log(awayRecord)

          const homeTeamInfo = await row
            .locator("td:nth-child(4) > div:nth-child(1)");
          const homeText = await homeTeamInfo
            .locator("> div:nth-child(1)")
            .textContent();
          const home = mapTeamName(homeText);
          const homeRecordText = await homeTeamInfo.locator("> div:nth-child(2)")
            .textContent();
          const { actual: homeRecord, smoothed: smoothedHomeRecord } = calculateWinningPercentage(homeRecordText);
  
          const spreadText = await row
            .locator("td:nth-child(3) > div:nth-child(2)")
            .textContent();
  
          let spreadMatch;
          if (spreadText === "EVEN") {
            spreadMatch = [null, homeText, 0];
          } else {
            const spreadRegex = /(\w+)\s(\-?\d+(?:\.5)?)/g;
            spreadMatch = spreadRegex.exec(spreadText);
          }
  
          const favorite = mapTeamName(spreadMatch[1]);
          // Spread is always negative, so flip it
          const spread = Math.abs(parseFloat(spreadMatch[2]));
  
          const stats = {
            away,
            awayRecord,
            smoothedAwayRecord,
            home,
            homeRecord,
            smoothedHomeRecord,
            favorite,
            spread,
            nflWeek,
            dayOfWeek,
            time
          };
  
          if (!useSpecificWeek) {
            const $finalScore = await row.locator(
              "td:nth-child(5) > div:nth-child(2)"
            );
            const finalScore = await $finalScore.textContent();
            const scoreRegex = /(\d+)\s\-\s(\d+)/g;
            const scoreMatch = scoreRegex.exec(finalScore);
            const score = {
              away: parseInt(scoreMatch[1]),
              home: parseInt(scoreMatch[2]),
            };
            stats.score = score;
    
            stats.is_favorite_home = stats.favorite === home ? 1 : 0;
    
            // if positive, home team won
            stats.faveWonBy = stats.favorite === home ? 
              score.home - score.away :
              score.away - score.home;

            if (stats.faveWonBy === spread) {
              console.log('spread matches pd')
            }

            stats.favoriteCovered = stats.faveWonBy > spread;
          }
  
          games.push(stats);
        }
      }
      weekResults[week] = games;
    }

    console.log("Saving weekly team stats to disk");
    try {
      const outputPath = useSpecificWeek ? 
        `./results/matchups/${year}-week-${weekNo}.json` :
        `./results/season-matchups/${year}.json`;

      const dataToSave = useSpecificWeek ? weekResults[date] : weekResults;

      writeFileSync(outputPath, JSON.stringify(dataToSave, null, 2), "utf8");
      console.log("Data successfully saved to disk", outputPath);
    } catch (error) {
      console.log("An error has occurred ", error);
    }
  });
}
