const teamMapping = {
  "Arizona Cardinals": "Arizona",
  ARI: "Arizona",
  "Atlanta Falcons": "Atlanta",
  ATL: "Atlanta",
  "Baltimore Ravens": "Baltimore",
  BAL: "Baltimore",
  "Buffalo Bills": "Buffalo",
  BUF: "Buffalo",
  "Carolina Panthers": "Carolina",
  CAR: "Carolina",
  "Chicago Bears": "Chicago",
  CHI: "Chicago",
  "Cincinnati Bengals": "Cincinnati",
  CIN: "Cincinnati",
  "Cleveland Browns": "Cleveland",
  CLE: "Cleveland",
  "Dallas Cowboys": "Dallas",
  DAL: "Dallas",
  "Denver Broncos": "Denver",
  DEN: "Denver",
  "Detroit Lions": "Detroit",
  DET: "Detroit",
  "Green Bay Packers": "Green Bay",
  GB: "Green Bay",
  "Houston Texans": "Houston",
  HOU: "Houston",
  "Indianapolis Colts": "Indianapolis",
  IND: "Indianapolis",
  "Jacksonville Jaguars": "Jacksonville",
  JAX: "Jacksonville",
  "Kansas City Chiefs": "Kansas City",
  KC: "Kansas City",
  OAK: "Las Vegas",
  "Oakland Raiders": "Las Vegas",
  "Las Vegas Raiders": "Las Vegas",
  LV: "Las Vegas",
  "Los Angeles Chargers": "LA Chargers",
  "San Diego Chargers": "LA Chargers",
  SD: "LA Chargers",
  LAC: "LA Chargers",
  "Los Angeles Rams": "LA Rams",
  "St Louis Rams": "LA Rams",
  STL: "LA Rams",
  LA: "LA Rams",
  LAR: "LA Rams",
  "Miami Dolphins": "Miami",
  MIA: "Miami",
  "Minnesota Vikings": "Minnesota",
  MIN: "Minnesota",
  "New England Patriots": "New England",
  NE: "New England",
  "New Orleans Saints": "New Orleans",
  NO: "New Orleans",
  "New York Giants": "NY Giants",
  NYG: "NY Giants",
  "New York Jets": "NY Jets",
  NYJ: "NY Jets",
  "Philadelphia Eagles": "Philadelphia",
  PHI: "Philadelphia",
  "Pittsburgh Steelers": "Pittsburgh",
  PIT: "Pittsburgh",
  "San Francisco 49ers": "San Francisco",
  SF: "San Francisco",
  "Seattle Seahawks": "Seattle",
  SEA: "Seattle",
  "Tampa Bay Buccaneers": "Tampa Bay",
  TB: "Tampa Bay",
  TEN: "Tennessee",
  "Tennessee Titans": "Tennessee",
  WAS: "Washington",
  "Washington Commanders": "Washington",
  "Washington Football Team": "Washington",
  "Washington Redskins": "Washington",
};

function mapTeamName(fullTeamName) {
  if (!teamMapping[fullTeamName]) {
    throw new Error(`No mapping for team ${fullTeamName}`);
  }

  return teamMapping[fullTeamName];
}

module.exports = mapTeamName;