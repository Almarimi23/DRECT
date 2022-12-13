// const ObjectsToCsv = require("objects-to-csv");
// const fs = require("fs");
// const parse = require("csv-parse/lib/sync");

// const log = console.log;

// const allMembers = [1000, 2000, 3000, 4000];

// let completedMembers = [];

// allMembers.forEach((item) => {
//   const chanDataLocation = `scraped_data member ${item}.csv`;
//   const allchanCsvData = fs.readFileSync(chanDataLocation).toString();

//   //calling the npm package and save to records
//   const chanRecords = parse(allchanCsvData, {
//     columns: true,
//     skip_empty_lines: true,
//   });
//   completedMembers = [...completedMembers, ...chanRecords];
// });

// log(completedMembers.length);

// // const chanDataLocation = "scraped_data challenges full.csv";
// // const allchanCsvData = fs.readFileSync(chanDataLocation).toString();

// // //calling the npm package and save to records
// // const chanRecords = parse(allchanCsvData, {
// //   columns: true,
// //   skip_empty_lines: true,
// // });

// // let group = [];

// // chanRecords.forEach((challenge, index) => {
// //   if (group.length.toString().endsWith("000")) {
// //     log("saving ....");
// //     new ObjectsToCsv(group).toDisk(`./scraped_data first ${index}.csv`, {
// //       allColumns: true,
// //     });
// //     group = [];
// //   }

// //   group.push(challenge);
// // });

// log("saving last part ....");
// new ObjectsToCsv(completedMembers).toDisk(`./scraped_data combined data.csv`, {
//   allColumns: true,
// });

const axios = require("axios");

(async () => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/challenges/?startDateEnd=2010-07-22&status=Completed&perPage=100&page=61&sortBy=startDate&sortOrder=desc&tracks[]=Dev&types[]=CH&types[]=F2F&types[]=TSK`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  console.log(data);
})();
