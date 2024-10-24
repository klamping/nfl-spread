const puppeteer = require('puppeteer');

async function scrapeTeamData(teamName, date) {
  const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    const ppgUrl = `https://www.teamrankings.com/nfl/stat/points-per-game?date=${date}`;
    console.log(ppgUrl);

  // Go to the target page
  await page.goto(ppgUrl, {
    waitUntil: "networkidle2",
  });
    
  // Wait for the page to load
  await page.waitForSelector('[role="main"]');

  // Scrape the data for the provided team
    const rows = await page.$$("#DataTables_Table_0 tbody tr");
    console.log(rows)

    let teamRow;
    for (let row of rows) {
        const rowInnerText = row.$('td:nth-child(1)').innerText;
        console.log(rowInnerText);
        if (rowInnerText.trim() === team) {
            teamRow = row;
            break;
        }
    }

    if (!teamRow) {
      throw new Error(`Team ${team} not found`);
    }

//   console.log(teamData);

  await browser.close();
}

scrapeTeamData('Kansas City', '2023-09-01')
  .then(() => {
    console.log('Scraping completed successfully.');
  })
  .catch((error) => {
    console.error(`Error occurred: ${error.message}`);
  });
